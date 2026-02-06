# 🎙️ AI-Generated Voice Detection (Multi-Language)

A production-ready **FastAPI** service that detects whether an uploaded audio clip is **human** or **AI-generated** using ML-based acoustic feature analysis (MFCCs, spectral features, ZCR) and an **XGBoost** classifier.

Built for hackathon deployment with real-time inference and public API access.

---

## 🚀 Live API

### Base URL

https://ai-generated-voice-detection-multi-gzld.onrender.com

---

### OUTPUT

<img width="1207" height="934" alt="Image" src="https://github.com/user-attachments/assets/5b811c30-daf4-4960-8b4f-f28adb5ec107" />

---

## 🔐 Authentication

All endpoints require an API key via header:

x-api-key: hackathon_voice_ai_2026

---

## 📌 Features

- Detects AI vs Human voice  
- Supports WAV / MP3 / MPEG audio  
- Automatic feature extraction (MFCC + spectral)  
- XGBoost classifier  
- FastAPI backend  
- Public cloud deployment (Render)  
- Multipart upload + Base64 support  
- Confidence score returned  
- Designed for multi-language expansion  

---

## 📡 API Endpoints

---

### 1️⃣ Upload Audio File (Recommended)

**POST**

/api/voice-detection-file

#### Headers

x-api-key: hackathon_voice_ai_2026

#### Body (multipart/form-data)

file: <audio file>

#### Example (curl)

```bash
curl -X POST \
https://ai-generated-voice-detection-multi-gzld.onrender.com/api/voice-detection-file \
-H "x-api-key: hackathon_voice_ai_2026" \
-F "file=@sample.wav"
```
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.98
}

---

### 2️⃣ Base64 Audio Endpoint

**POST**

/api/voice-detection

---

#### Headers

x-api-key: hackathon_voice_ai_2026

---

#### Body (multipart/form-data)
{
  "language": "English",
  "audioFormat": "mpeg",
  "audioBase64": "<BASE64 STRING>"
}

---

### 🧠 Model Pipeline

1. Audio normalization (16kHz mono)

2. Feature extraction:

    -> MFCC mean + std

    -> Spectral centroid

    -> Spectral rolloff

    -> Zero Crossing Rate

3. Standard scaling

4. XGBoost classification

5. Probability → Confidence score

---

### 🗂️ Project Structure

AI-Generated-Voice-Detection/
│
├── LANGUAGES/
│   ├── api_eng.py
│   ├── feature_extractor_eng.py
│   ├── model/
│   │   ├── voice_detector_NEW.pkl
│   │   └── scaler_NEW.pkl
│
├── requirements.txt
└── README.md

---

### ⚙️ Local Setup

git clone <repo>
cd AI-Generated-Voice-Detection
pip install -r requirements.txt
uvicorn LANGUAGES.api_eng:app --reload

---

#### Server runs at:  http://127.0.0.1:8000
#### Swagger UI: http://127.0.0.1:8000/docs

---

## 🧪 Model Training Summary

- Features: MFCC + spectral  
- Dataset: Mixed human + AI voices  
- Classifier: XGBoost  
- ROC AUC ≈ 0.99+  
- Real-time inference capable  

---

## 🛠 Tech Stack

- Python  
- FastAPI  
- Librosa  
- XGBoost  
- Scikit-learn  
- Joblib  
- Uvicorn  
- Render  

---

## ⚠️ Notes

- Confidence reflects model probability, not absolute certainty.  
- Designed for hackathon deployment and extensibility.  
- Multi-language models can be plugged into the same pipeline.  

---

## 👨‍💻 Author(Full-code)

Aryan  
Hackathon Project – 2026 

---

## 👨‍💻 Co-Author(Dataset)

Atharsh Bharathkumar  
Hackathon Project – 2026 

---

## 📜 License

MIT  

---

### ✅ After pasting

```bash
git add README.md
git commit -m "Add professional README"
git push
```
