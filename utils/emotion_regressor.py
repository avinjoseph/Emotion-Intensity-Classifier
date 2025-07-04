import torch.nn as nn
from transformers import DistilBertModel

# EmotionRegressor is a custom model that uses DistilBERT for emotion regression tasks.
# It inherits from nn.Module and implements the forward method to process input text and output emotion scores
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
