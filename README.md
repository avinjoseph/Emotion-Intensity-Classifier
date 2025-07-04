# Emotion-Intensity-Classifier
Emotion Intensity Mutli label Classification
## Folder Structure

```
Emotion-Intensity-Classifier/
├── data/                           # Datasets 
|   |-- dataset_augmented           # Augmented data
|   |-- test_data                   # Test data is a validation data for random testing.
|   |-- track-b                     # Original datas
├── notebooks/                      # Jupyter notebooks for experiments and analysis
|    ├── models/                    # Model architectures and saved models
|           |-- emotion_classifier  # DistilBert Model
|    |-- analysis.ipynb             # Analysis of the dataset
|    |-- emotion_classifier.ipynb   # Source code for training and evaluation
├── utils/                          # Utility scripts and helper functions
|    |-- data_preprocessing         # preprocessing of the dataset
|    |-- emotion_regressor          # Class file 
|-- main.py                         # Main file which contains the predict function to pass the file and generate output
├── requirements.txt                # Lists all Python dependencies required to run the project
├── README.md                       # Main project documentation and usage instructions
```
