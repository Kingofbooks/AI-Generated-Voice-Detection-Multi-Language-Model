import torch
import librosa
import numpy as np
from torchvision import models
import torch.nn as nn
SAMPLE_RATE=16000
N_MELS=64
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def load_audio(path):
    y,_=librosa.load(path,sr=SAMPLE_RATE)
    if len(y)<SAMPLE_RATE*2:
        y=np.pad(y,(0,SAMPLE_RATE*2-len(y)))
    return y[:SAMPLE_RATE*2]

def make_mel(audio):
    mel=librosa.feature.melspectrogram(y=audio,sr=SAMPLE_RATE,n_mels=N_MELS)
    mel=librosa.power_to_db(mel)
    mel=(mel-mel.mean())/(mel.std()+1e-6)
    return torch.tensor(mel).unsqueeze(0).unsqueeze(0)

model=models.resnet18(weights=None)
model.conv1=nn.Conv2d(1,64,7,2,3,bias=False)
model.fc=nn.Linear(512,1)
model.load_state_dict(torch.load("E:\AI-Generated Voice Detection (Multi-Language)\AI-Generated-Voice-Detection\ENGLISH\model\scaler.pkl",map_location=DEVICE))
model.to(DEVICE)
model.eval()

audio=load_audio("E:\\AI-Generated Voice Detection (Multi-Language)\\sample\\ai01.mpeg")
mel=make_mel(audio).to(DEVICE)

with torch.no_grad():
    p=torch.sigmoid(model(mel)).item()

print("AI probability:\n",p)
print("Pred:", "AI" if p > 0.5 else "HUMAN")

