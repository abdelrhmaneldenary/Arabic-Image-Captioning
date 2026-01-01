# ==========================================
# 🚀 FINAL MASTER CELL: FIXED WARNINGS + DATASET NAMES
# ==========================================
import os

# --- 1. FIX THE WARNING (Must be at the very top) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Libraries for Proper Arabic Display
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    print("⚠️ Arabic libraries not found. Text will be reversed manually.")

# 0. CONFIGURATION
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_FOLDER = '/kaggle/input/arabic-flickr8k-dataset/Images'
CAPTION_FILE = '/kaggle/input/arabic-flickr8k-dataset/captions.txt'

# Hyperparameters (SAFE MODE)
EMBED_DIM = 768
ATTENTION_DIM = 512
DECODER_DIM = 512
ENCODER_DIM = 2048
DROPOUT = 0.3
LEARNING_RATE = 3e-4
PATIENCE = 4            # Wait 4 epochs for BLEU to improve
BEAM_ALPHA = 0.8        # Length Penalty (Higher = Longer sentences)

# 1. DATA PREPARATION
# ---------------------------------------------------------
def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if len(line) < 2: continue
        parts = line.split('\t')
        if len(parts) < 2: parts = line.split(' ', 1)
        if len(parts) == 2:
            data.append([parts[0].split('#')[0], parts[1].strip()])
    return pd.DataFrame(data, columns=['image', 'caption'])

print("⏳ Loading Data & Building Vocab...")
df = load_data(CAPTION_FILE)

# BERT Tokenizer
raw_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
bert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
full_embedding = bert_model.embeddings.word_embeddings.weight

# Filter Vocab
used_ids = set([raw_tokenizer.cls_token_id, raw_tokenizer.sep_token_id, raw_tokenizer.pad_token_id, raw_tokenizer.unk_token_id])
for cap in tqdm(df['caption']):
    used_ids.update(raw_tokenizer.encode(cap, add_special_tokens=False))

compact_ids = sorted(list(used_ids))
c2o = {i: old_id for i, old_id in enumerate(compact_ids)}
o2c = {old_id: i for i, old_id in enumerate(compact_ids)}
new_matrix = full_embedding[compact_ids].clone().detach()

class CompactVocabularyBPE:
    def __init__(self, tokenizer, c2o, o2c):
        self.tokenizer = tokenizer; self.to_new = o2c; self.to_old = c2o
        self.pad_token_id = o2c[tokenizer.pad_token_id]
        self.cls_token_id = o2c[tokenizer.cls_token_id]
        self.sep_token_id = o2c[tokenizer.sep_token_id]
    def numericalize(self, text):
        return [self.to_new.get(t, self.to_new[self.tokenizer.unk_token_id]) for t in self.tokenizer.encode(text)]
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.tolist()
        return self.tokenizer.decode([self.to_old.get(t, self.tokenizer.unk_token_id) for t in token_ids], skip_special_tokens=True)
    def __len__(self): return len(self.to_new)

vocab = CompactVocabularyBPE(raw_tokenizer, c2o, o2c)

# Splits
train_imgs, val_imgs = train_test_split(df['image'].unique(), test_size=0.1, random_state=42)
train_df = df[df['image'].isin(train_imgs)].reset_index(drop=True)
val_df = df[df['image'].isin(val_imgs)].reset_index(drop=True)

# Transforms (Deterministic)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class FlickrDataset(Dataset):
    def __init__(self, root, df, vocab, transform):
        self.root = root; self.df = df; self.vocab = vocab; self.transform = transform
        self.imgs = df['image'].tolist(); self.caps = df['caption'].tolist()
    def __len__(self): return len(self.caps)
    def __getitem__(self, idx):
        try: img = Image.open(os.path.join(self.root, self.imgs[idx])).convert("RGB")
        except: img = Image.new('RGB', (224, 224), (0, 0, 0))
        return self.transform(img), torch.tensor(self.vocab.numericalize(self.caps[idx]), dtype=torch.long)

class CollateAPI:
    def __init__(self, pad_idx): self.pad_idx = pad_idx
    def __call__(self, batch):
        imgs = torch.stack([i[0] for i in batch], 0)
        caps = pad_sequence([i[1] for i in batch], batch_first=True, padding_value=self.pad_idx)
        lens = torch.tensor([len(i[1]) for i in batch])
        return imgs, caps, lens

# --- DEFINE DATASETS (Crucial for visualization) ---
train_ds = FlickrDataset(IMG_FOLDER, train_df, vocab, transform)
val_ds = FlickrDataset(IMG_FOLDER, val_df, vocab, transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=CollateAPI(vocab.pad_token_id), num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=CollateAPI(vocab.pad_token_id), num_workers=2)

# 2. MODEL DEFINITIONS (FROZEN ENCODER)
# ---------------------------------------------------------
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights='ResNet101_Weights.DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        for p in self.resnet.parameters(): p.requires_grad = False
    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        return out.permute(0, 2, 3, 1)

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU(); self.softmax = nn.Softmax(dim=1)
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out); att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        return (encoder_out * alpha.unsqueeze(2)).sum(dim=1), alpha

class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.3):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim); self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim); self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
    def load_embeddings(self, weights):
        self.embedding.weight = nn.Parameter(weights); self.embedding.weight.requires_grad = True
    def init_hidden_state(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0); encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]; encoded_captions = encoded_captions[sort_ind]
        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            awe, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], gate * awe], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            predictions[:batch_size_t, t, :] = self.fc(self.dropout(h))
            alphas[:batch_size_t, t, :] = alpha
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

# 3. HELPER FUNCTIONS (PENALIZED BEAM SEARCH)
# ---------------------------------------------------------
def beam_search_penalized(encoder_out, decoder, vocab, beam_size=3, max_len=20, alpha=0.0):
    k = beam_size; vocab_size = len(vocab)
    encoder_out = encoder_out.view(1, -1, 2048).expand(k, 196, 2048)
    k_prev_words = torch.LongTensor([[vocab.cls_token_id]] * k).to(device)
    seqs = k_prev_words; top_k_scores = torch.zeros(k, 1).to(device)
    complete_seqs = []; complete_seqs_scores = []
    h, c = decoder.init_hidden_state(encoder_out)
    step = 1
    while step < max_len:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        awe, _ = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        h, c = decoder.decode_step(torch.cat([embeddings, gate * awe], dim=1), (h, c))
        
        scores = F.log_softmax(decoder.fc(h), dim=1)
        scores = top_k_scores.expand_as(scores) + scores
        
        if step == 1: top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else: top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        
        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size
        
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab.sep_token_id]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        if len(complete_inds) > 0:
            for idx in complete_inds:
                length = step + 1
                final_score = top_k_scores[idx].item() / (length ** alpha)
                complete_seqs.append(seqs[idx].tolist())
                complete_seqs_scores.append(final_score)
                
        k -= len(complete_inds)
        if k == 0: break
        
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        step += 1
        
    if len(complete_seqs) == 0: return vocab.decode(seqs[0])
    best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
    return vocab.decode(complete_seqs[best_idx])

def visualize_random_sample(encoder, decoder, val_ds, vocab):
    encoder.eval(); decoder.eval()
    idx = random.randint(0, len(val_ds) - 1)
    img_tensor, _ = val_ds[idx]
    
    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(device)
        features = encoder(img_input)
        caption = beam_search_penalized(features, decoder, vocab, beam_size=5, alpha=BEAM_ALPHA)
    
    if ARABIC_SUPPORT:
        reshaped_text = arabic_reshaper.reshape(caption)
        display_text = get_display(reshaped_text)
    else:
        display_text = caption[::-1]

    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_display)
    plt.title(f"Pred: {display_text}", fontsize=16, color='blue', loc='center')
    plt.axis('off')
    plt.show()

def evaluate_bleu(encoder, decoder, val_df, beam_size):
    encoder.eval(); decoder.eval()
    sample_imgs = val_df['image'].unique()[:400]
    image_to_captions = val_df[val_df['image'].isin(sample_imgs)].groupby('image')['caption'].apply(list).to_dict()
    
    references, hypotheses = [], []
    for image_name, caption_list in tqdm(image_to_captions.items(), desc=f"Calculating BLEU", leave=False):
        try:
            image = Image.open(os.path.join(IMG_FOLDER, image_name)).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
        except: continue
        with torch.no_grad():
            pred = beam_search_penalized(encoder(image_tensor), decoder, vocab, beam_size, alpha=BEAM_ALPHA)
            hypotheses.append(pred.split())
            references.append([c.split() for c in caption_list])
    return corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method4)

def train_one_epoch(loader, encoder, decoder, criterion, optimizer, epoch):
    decoder.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for imgs, caps, lens in loop:
        imgs, caps, lens = imgs.to(device), caps.to(device), lens.to(device)
        feats = encoder(imgs)
        preds, caps_sorted, decode_lens, _, _ = decoder(feats, caps, lens)
        targets = pack_padded_sequence(caps_sorted[:, 1:], decode_lens, batch_first=True).data
        preds = pack_padded_sequence(preds, decode_lens, batch_first=True).data
        loss = criterion(preds, targets)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0); optimizer.step()
        total_loss += loss.item(); loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

# 4. INITIALIZATION
# ---------------------------------------------------------
print("🏗️ Initializing Models...")
encoder = EncoderCNN().to(device)
decoder = DecoderRNN(ATTENTION_DIM, EMBED_DIM, DECODER_DIM, len(vocab), ENCODER_DIM, DROPOUT)
decoder.load_embeddings(new_matrix)
decoder = decoder.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)
optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

# 5. TRAINING LOOP (BLEU-BASED STOPPING)
# ---------------------------------------------------------
print("🚀 Starting Training (Monitor: BLEU Score)...")
best_bleu = 0.0
patience_counter = 0

for epoch in range(1, 21): 
    train_loss = train_one_epoch(train_loader, encoder, decoder, criterion, optimizer, epoch)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    visualize_random_sample(encoder, decoder, val_ds, vocab)

    bleu = evaluate_bleu(encoder, decoder, val_df, beam_size=5)
    print(f"   🔵 BLEU-4: {bleu*100:.2f}")

    if bleu > best_bleu:
        best_bleu = bleu
        patience_counter = 0
        torch.save(decoder.state_dict(), "best_model_bleu.pth")
        torch.save(encoder.state_dict(), "best_encoder_bleu.pth")
        print("   ⭐ BLEU Improved! Saving Best Model.")
    else:
        patience_counter += 1
        print(f"   ⚠️ No BLEU Improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"🛑 Early Stopping! Best BLEU was: {best_bleu*100:.2f}")
            break

print("✅ Training Done.")