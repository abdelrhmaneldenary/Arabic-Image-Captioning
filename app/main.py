import torch
import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
# This import will now work because you created the file above!
from app.model_utils import EncoderCNN, DecoderRNN, CompactVocabularyBPE
import torchvision.transforms as transforms
from transformers import AutoTokenizer

# --- 1. SETUP & CONFIGURATION ---
app = FastAPI(title="Arabic Caption AI")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_data = {}

# --- 2. HTML INTERFACE ---
html_content = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic Image Captioning</title>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Cairo', sans-serif; background-color: #f4f7f6; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; flex-direction: column; }
        .container { background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); text-align: center; width: 90%; max-width: 500px; }
        h1 { color: #333; margin-bottom: 30px; }
        .file-upload { position: relative; display: inline-block; margin-bottom: 20px; }
        .file-upload input[type="file"] { display: none; }
        .custom-upload-btn { background: #4a90e2; color: white; padding: 15px 30px; border-radius: 50px; cursor: pointer; font-size: 18px; transition: 0.3s; display: inline-block; }
        .custom-upload-btn:hover { background: #357abd; }
        #preview-image { max-width: 100%; max-height: 300px; border-radius: 10px; margin-top: 20px; display: none; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        #result-box { margin-top: 25px; font-size: 24px; font-weight: bold; color: #2c3e50; direction: rtl; min-height: 40px; }
        .loading { color: #888; font-size: 18px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 قارئ الصور الذكي</h1>
        <div class="file-upload">
            <label for="file-input" class="custom-upload-btn">📂 اختر صورة</label>
            <input id="file-input" type="file" accept="image/*" onchange="previewAndPredict(event)">
        </div>
        <img id="preview-image" src="" alt="Preview">
        <div id="result-box"></div>
    </div>
    <script>
        async function previewAndPredict(event) {
            const file = event.target.files[0];
            if (!file) return;
            const imgElement = document.getElementById('preview-image');
            imgElement.src = URL.createObjectURL(file);
            imgElement.style.display = 'block';
            const resultBox = document.getElementById('result-box');
            resultBox.innerHTML = '<span class="loading">... جاري التحليل</span>';
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                resultBox.innerText = data.caption_arabic;
            } catch (error) {
                console.error(error);
                resultBox.innerText = "حدث خطأ في الاتصال";
            }
        }
    </script>
</body>
</html>
"""

# --- 3. STARTUP EVENT (Robust Load) ---
@app.on_event("startup")
def load_model():
    global model_data
    print("⏳ Loading Model Bundle...")
    
    # Check if file exists first
    if not os.path.exists("model/arabic_captioning_bundle.pth"):
        print("❌ ERROR: 'model/arabic_captioning_bundle.pth' not found in the 'model' folder!")
        return

    try:
        # Load Checkpoint
        checkpoint = torch.load("model/arabic_captioning_bundle.pth", map_location=DEVICE)
        
        # Rebuild Vocab
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        vocab = CompactVocabularyBPE(tokenizer, checkpoint['c2o'], checkpoint['o2c'])
        
        # Initialize Models
        encoder = EncoderCNN().to(DEVICE)
        decoder = DecoderRNN(
            attention_dim=512, embed_dim=768, decoder_dim=512,
            vocab_size=len(vocab), encoder_dim=2048, dropout=0.5
        ).to(DEVICE)
        
        # Load Weights
        encoder.load_state_dict(checkpoint['encoder_state'])
        decoder.load_state_dict(checkpoint['decoder_state'])
        
        encoder.eval()
        decoder.eval()
        
        model_data['encoder'] = encoder
        model_data['decoder'] = decoder
        model_data['vocab'] = vocab
        
        print("✅ Model Loaded Successfully!")
        
    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")

# --- 4. PREDICTION LOGIC ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def generate_caption(image, max_len=20):
    if 'encoder' not in model_data:
        return "⚠️ Error: Model not loaded. Check server terminal."

    encoder = model_data['encoder']
    decoder = model_data['decoder']
    vocab = model_data['vocab']
    
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        encoder_out = encoder(img_tensor)
        encoder_out = encoder_out.view(1, -1, 2048)
        
        h, c = decoder.init_hidden_state(encoder_out)
        word = torch.LongTensor([[vocab.cls_token_id]]).to(DEVICE)
        
        sentence = []
        for _ in range(max_len):
            embeddings = decoder.embedding(word).squeeze(1)
            awe, _ = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            h, c = decoder.decode_step(torch.cat([embeddings, gate * awe], dim=1), (h, c))
            
            scores = decoder.fc(h)
            best_idx = scores.argmax(dim=1).item()
            
            if best_idx == vocab.sep_token_id:
                break
                
            sentence.append(best_idx)
            word = torch.LongTensor([[best_idx]]).to(DEVICE)
            
    return vocab.decode(sentence)

# --- 5. ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return html_content

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    caption = generate_caption(image)
    return {"caption_arabic": caption}