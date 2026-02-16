# 📸 Arabic Image Captioning (قارئ الصور الذكي)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Abdelrhman/arabic-image-captioning)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Bridging the gap between Computer Vision and Arabic NLP: From LSTMs to Transformers."**

## 🚀 Project Overview
This project is a production-ready **Deep Learning system** capable of generating descriptive, grammatically correct **Arabic captions** for images. 

I have evolved this system through two major architectural generations, moving from traditional CNN-RNN structures to a state-of-the-art **Vision Transformer (ViT) + GPT-2** pipeline. This progress reflects my commitment to engineering excellence and staying at the forefront of AI research.

### 🌟 Try It Live
Click the badge above or visit the link below to test the model:
👉 **[Live Demo on Hugging Face](https://huggingface.co/spaces/Abdlerhman/VLM)**

---

## 🏆 Performance & Architectural Evolution
By pivoting to a Transformer architecture and scaling the training data through a localized **Flickr30k-Arabic** corpus, I achieved a massive leap in performance.

| Metric | v1.0 (CNN-RNN) | **v2.0 (Transformer)** | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Visual Encoder** | ResNet-101 (CNN) | **ViT-Base (Transformer)** | Better global context |
| **Language Decoder** | LSTM (RNN) | **AraGPT2 (Transformer)** | Pre-trained Arabic fluency |
| **Dataset** | Flickr8k (~8,000) | **Flickr30k + 8k (~39,000)** | 5x Data Diversity |
| **BLEU-4 Score** | 14.09 | **25.00** | **+77% Accuracy Gain** |

---

## 🧠 Technical Architecture (v2.0 - Current)
The current system utilizes a **VisionEncoderDecoder** framework, merging high-fidelity visual features with generative linguistic knowledge.



### 1. Visual Encoder: Vision Transformer (ViT)
* **Architecture:** `google/vit-base-patch16-224-in21k`.
* **Mechanism:** Instead of traditional convolutions, the image is flattened into fixed-size patches and processed via **Self-Attention**, capturing long-range spatial dependencies that CNNs often miss.

### 2. Language Decoder: AraGPT2
* **Architecture:** `aubmindlab/aragpt2-base`.
* **BOS Injection:** I engineered a custom tokenization bridge by injecting a **Beginning-of-Sentence [BOS]** token, solving the "cold start" issue common in GPT-based decoders for multimodal tasks.

### 3. Training & Optimization
* **Warm-Start Fine-Tuning:** Initialized with pre-trained weights to leverage prior knowledge of objects and Arabic grammar.
* **Squeeze Phase:** Implemented a two-stage training schedule, dropping the Learning Rate by 10x in the final epochs to ensure precise convergence.
* **Beam Search (k=5):** Used during inference to explore multiple word sequences, ensuring the final caption is the most probable and descriptive.

---

## 📜 Legacy Architecture (v1.0)
For transparency and educational purposes, the initial version of this project used:
* **Encoder:** ResNet-101 with the final layers removed to extract a 14x14 spatial feature grid.
* **Decoder:** A custom LSTM with **Soft Attention** (Show, Attend, and Tell).
* **Tokenizer:** Byte Pair Encoding (BPE) via AraBERT.

---

## 🛠️ Tech Stack & Tools
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Computer Vision:** ViT, ResNet
* **NLP:** AraGPT2, AraBERT, NLTK, MarianMT (for data translation)
* **Backend:** FastAPI (Asynchronous API)
* **Deployment:** Docker, Hugging Face Spaces

---

## 💻 How to Run Locally

### Option 1: Using Docker (Recommended)
```bash
# 1. Clone the repository
git clone [https://github.com/Abdelrhman/Arabic-Image-Captioning.git](https://github.com/Abdelrhman/Arabic-Image-Captioning.git)
cd Arabic-Image-Captioning

# 2. Build the Docker Image
docker build -t arabic-caption-v2 .

# 3. Run the Container
docker run -p 8000:8000 arabic-caption-v2
