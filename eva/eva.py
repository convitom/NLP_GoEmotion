import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.train import EncoderForMultiLabelClassification
from src.dataloader import BACKBONE_REGISTRY, EMOTION_NAMES

TEST_PATH = r"D:\USTH\nlp\final_prj\data\test.csv"
OUT_PATH = r"D:\USTH\nlp\final_prj\eva\test_predictions.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load test data
df = pd.read_csv(TEST_PATH)

texts = df["text"].tolist()
true_labels = df["labels"].tolist()  # dạng list hoặc string




CKPT_PATH = r"C:\Users\admin\Downloads\best (1).pth"

checkpoint = torch.load(CKPT_PATH, map_location="cpu")
cfg = checkpoint["cfg"]

name = cfg["model"]["name"]
pretrained = BACKBONE_REGISTRY[name]["pretrained"]

tokenizer = AutoTokenizer.from_pretrained(pretrained)

model = EncoderForMultiLabelClassification(
    pretrained_name=pretrained,
    num_labels=cfg["data"]["num_emotions"]
)

model.load_state_dict(checkpoint["model_state"])
model.eval()

def vector_to_labels(vec):
    labels = [EMOTION_NAMES[i] for i, v in enumerate(vec) if v == 1]
    return labels

def labels_to_vector(label_str):
    labels = ast.literal_eval(label_str)   # "['annoyance','disapproval']"
    vec = np.zeros(len(EMOTION_NAMES), dtype=int)

    for l in labels:
        if l in EMOTION_NAMES:
            idx = EMOTION_NAMES.index(l)
            vec[idx] = 1

    return vec

def predict_batch(texts, batch_size=32):

    preds = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        probs = torch.sigmoid(logits)

        preds.extend(probs.cpu().numpy())

    return np.array(preds)

pred_probs = predict_batch(texts)

threshold = 0.1
pred_labels = (pred_probs > threshold).astype(int)

df["pred_label_name"] = [
    vector_to_labels(v) for v in pred_labels
]

from sklearn.metrics import f1_score, precision_score, recall_score

y_true = np.vstack(df["label_name"].apply(labels_to_vector))
y_pred = pred_labels

micro_f1 = f1_score(y_true, y_pred, average="micro")
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

precision = precision_score(y_true, y_pred, average="micro")
recall = recall_score(y_true, y_pred, average="micro")

with open("metrics.json","w") as f:
    f.write(f"micro_f1: {micro_f1:.4f}\n")
    f.write(f"macro_f1: {macro_f1:.4f}\n")
    f.write(f"weighted_f1: {weighted_f1:.4f}\n")
    f.write(f"precision: {precision:.4f}\n")
    f.write(f"recall: {recall:.4f}\n")
    
df.to_csv(OUT_PATH, index=False)

