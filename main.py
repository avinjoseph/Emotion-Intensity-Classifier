import torch
import pandas as pd
import os
import sys
from transformers import DistilBertTokenizerFast, DistilBertModel
sys.path.append(os.path.abspath('..'))
import torch.nn as nn

from utils.data_preprocessing import DataPreprocessor# Assuming you have a data preprocessing module

class EmotionRegressor(nn.Module):
    def __init__(self, dropout=0.5, num_emotions=5):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.regressor(pooled)


def preprocess_texts(texts, tokenizer, max_len=64):
    # Join tokens if your CSV stores token lists
    # If texts are strings, skip the join step

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return encodings

def score_to_intensity(score):
    score = max(0, score)  # clip negatives to zero
    if score <= 0.25:
        return 0
    elif score <= 0.5 and score > 0.25:
        return 1
    elif score <= 0.75 and score > 0.5:
        return 2
    else:
        return 3

def predict(csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = EmotionRegressor()
    model.load_state_dict(torch.load("./notebooks/model/emotion_classifier_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Load data
    df = DataPreprocessor(csv_path)
    df.preprocess() 
    data = df.data

    # Preprocess text column (adjust column name if needed)
    encodings = preprocess_texts(data['text'].to_list(), tokenizer)

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = outputs.cpu().numpy()
    
    # intensity_preds = []
    # for sample_scores in preds:
    #     intensity_sample = [score_to_intensity(score) for score in sample_scores]
    #     intensity_preds.append(intensity_sample)
    
    return preds


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python main.py <path_to_csv>")
    #     sys.exit(1)
    csv_path = "./data/track-b.csv" # Replace with your actual CSV file path
    predictions = predict(csv_path)
    print(predictions)  # This will print the predictions for each emotion
    
    
    

