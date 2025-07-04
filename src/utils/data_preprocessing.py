import pandas as pd
import re
import os

class DataPreprocessor:
    def __init__(self, input_data):
        if isinstance(input_data, str) and os.path.exists(input_data):
            self.data = pd.read_csv(input_data)
        elif isinstance(input_data, str):
            self.data = pd.DataFrame({'text': [input_data]})
        elif isinstance(input_data, pd.DataFrame):
            self.data = input_data
        else:
            raise ValueError("Input must be a file path or a pandas DataFrame.")
        
    def lower_case(self, column = "text"):
        if column in self.data.columns:
            self.data[column] = self.data[column].astype(str).str.lower()
            
            

    #! Stop word removal can sometimes make it harder to capture the full emotion or meaning of the sentence
    # def remove_stopwords(self, column):
    #     # Remove stopwords from the specified column
    #     if column in self.data.columns:
    #         stop_words = set(stopwords.words('english'))
    #         self.data[column] = self.data[column].astype(str).apply(
    #             lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words])
    #         )

    def tokenize(self, column="text"):
        if column in self.data.columns:
            self.data[column] = self.data[column].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))


    def clean_text(self, column = "text"):
        if column in self.data.columns:
            self.data[column] = self.data[column].astype(str).apply(
                lambda x: re.sub(r'[^\w\s]', '', x) 
            ).str.replace(r'\s+',' ', regex=True).str.strip()

    def preprocess(self):
        self.lower_case()
        self.clean_text()
        self.tokenize()
    
    
    
