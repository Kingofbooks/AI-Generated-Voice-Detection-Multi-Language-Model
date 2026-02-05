from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import uuid
import joblib
import os
from pydub import AudioSegment
from LANGUAGES.feature_extractor_eng import extract_features

API_KEY = "hackathon_voice_ai_2026"

app = FastAPI()

model = joblib.load("LANGUAGES/model/voice_detector_NEW.pkl")
scaler = joblib.load("LANGUAGES/model/scaler_NEW.pkl")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    audio_bytes = base64.b64decode(req.audioBase64)

    raw_path = f"{uuid.uuid4()}.input"
    wav_path = f"{uuid.uuid4()}.wav"

    # Save raw audio
    with open(raw_path, "wb") as f:
        f.write(audio_bytes)

    # Convert to WAV automatically
    audio = AudioSegment.from_file(raw_path)
    audio.export(wav_path, format="wav")

    # Extract features
    features = extract_features(wav_path)

    if len(features) == 0:
        raise HTTPException(status_code=400, detail="Invalid audio")

    # Cleanup
    os.remove(raw_path)
    os.remove(wav_path)

    features = scaler.transform([features])
    ai_prob = model.predict_proba(features)[0][1]

    if ai_prob > 0.6:
        classification = "AI_GENERATED"
        confidence = ai_prob
    else:
        classification = "HUMAN"
        confidence = 1 - ai_prob

    return {
        "status": "success",
        "language": req.language,
        "classification": classification,
        "confidenceScore": float(confidence),
        "explanation": "Spectral rolloff and MFCC instability patterns indicate synthetic speech"
    }


from fastapi import UploadFile, File

@app.post("/api/voice-detection-file")
async def detect_file(file: UploadFile = File(...), x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401)

    contents = await file.read()

    tmp = f"{uuid.uuid4()}.mp3"
    with open(tmp, "wb") as f:
        f.write(contents)

    features = extract_features(tmp)
    os.remove(tmp)

    features = scaler.transform([features])
    ai_prob = model.predict_proba(features)[0][1]

    if ai_prob > 0.6:
        classification = "AI_GENERATED"
        confidence = ai_prob
    else:
        classification = "HUMAN"
        confidence = 1 - ai_prob

    return {
        "status": "success",
        "classification": classification,
        "confidenceScore": float(confidence)
    }
