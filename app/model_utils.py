import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. VOCABULARY CLASS ---
class CompactVocabularyBPE:
    def __init__(self, tokenizer, c2o, o2c):
        self.tokenizer = tokenizer
        self.to_new = o2c
        self.to_old = c2o
        self.pad_token_id = o2c[tokenizer.pad_token_id]
        self.cls_token_id = o2c[tokenizer.cls_token_id]
        self.sep_token_id = o2c[tokenizer.sep_token_id]

    def numericalize(self, text):
        return [self.to_new.get(t, self.to_new[self.tokenizer.unk_token_id]) for t in self.tokenizer.encode(text)]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.tolist()
        return self.tokenizer.decode([self.to_old.get(t, self.tokenizer.unk_token_id) for t in token_ids], skip_special_tokens=True)
    
    def __len__(self): return len(self.to_new)

# --- 2. ENCODER (ResNet) ---
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # Load ResNet
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights='ResNet101_Weights.DEFAULT')
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.fine_tune(False)

    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        return out.permute(0, 2, 3, 1)

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters(): p.requires_grad = False
        if fine_tune:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters(): p.requires_grad = True

# --- 3. ATTENTION ---
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        return (encoder_out * alpha.unsqueeze(2)).sum(dim=1), alpha

# --- 4. DECODER (LSTM) ---
class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def load_embeddings(self, weights):
        self.embedding.weight = nn.Parameter(weights)
        self.embedding.weight.requires_grad = True

    def init_hidden_state(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        # We generally don't use forward() in inference (we use the manual loop in main.py)
        # But we keep it here just in case.
        pass