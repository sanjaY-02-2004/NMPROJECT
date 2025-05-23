# Import necessary libraries
import pandas as pd
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load the dataset
def load_data(file_path):
    # Assuming the dataset is in CSV format with 'text' and 'entities' columns
    df = pd.read_csv(file_path)
    return df

# Preprocess the data for SpaCy
def preprocess_spacy(df):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    
    for index, row in df.iterrows():
        doc = nlp(row['text'])
        # Add entities to the doc
        for start, end, label in row['entities']:  # Assuming entities are in (start, end, label) format
            doc.ents += ((doc.char_span(start, end, label=label),))
        doc_bin.add(doc)
    
    doc_bin.to_disk("financial_docs.spacy")

# Define a custom dataset for Fine-tuned FinBERT
class FinancialDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return encoding, label

# Train the Fine-tuned FinBERT model
def train_fine_tuned_finbert(train_texts, train_labels):
    model = BertForTokenClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=len(set(train_labels)))
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=FinancialDataset(train_texts, train_labels),
    )
    
    trainer.train()

# Main function to execute the NER pipeline
def main():
    # Load and preprocess data
    df = load_data('c:\Users\paul\Downloads\FNER_data_train.csv')
    preprocess_spacy(df)
    
    # Split data for training and testing
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['entities'], test_size=0.2)
    
    # Train SpaCy NER model
    nlp = spacy.load("en_core_web_sm")
    # Add your training code for SpaCy NER here
    
    # Train Fine-tuned FinBERT model
    train_fine_tuned_finbert(train_texts, train_labels)

if __name__ == "__main__":
    main()
