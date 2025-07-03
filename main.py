import torch
import numpy as np
import os
import sys
from transformers import DistilBertTokenizerFast, DistilBertModel
sys.path.append(os.path.abspath('..'))
import torch.nn as nn
from utils.emotion_regressor import EmotionRegressor  
from utils.data_preprocessing import DataPreprocessor# Assuming you have a data preprocessing module



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
    score = max(0, score)  # clip negatives to 0
    if score <= 0.5:
        return 0
    elif score > 0.5 and score <= 1.5:
        return 1
    elif score > 1.5 and score <= 2.5:
        return 2
    else:
        return 3

def predict(csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = EmotionRegressor()
    model.load_state_dict(torch.load("./notebooks/model/emotion_classifier_model_2.pt", map_location=device))
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
    
    intensity_preds = []
    for sample_scores in preds:
        intensity_sample = [score_to_intensity(score) for score in sample_scores]
        intensity_preds.append(intensity_sample)
    
    return intensity_preds


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python main.py <path_to_csv>")
    #     sys.exit(1)
    csv_path = "./data/test_data.csv" # Replace with your actual CSV file path
    predictions = predict(csv_path)
    print(predictions)  # This will print the predictions for each emotion
    
    
    

