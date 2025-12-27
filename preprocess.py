import pandas as pd
import spacy
from collections import Counter
import torch
from torchtext.vocab import vocab

# Load the English tokenizer from spaCy
nlp = spacy.load("en_core_web_sm")

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df["story"].tolist()

def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        doc = nlp(sentence)
        counter.update([token.text.lower() for token in doc if token.is_alpha])
    return vocab(counter, specials=["<pad>", "<unk>"])

def numericalize_data(sentences, vocab):
    return [[vocab[token] for token in nlp(sentence) if token.is_alpha] for sentence in sentences]

if __name__ == "__main__":
    filepath = "C:\\Users\\Shashank S M\\Desktop\\storyai\\dataset_2016.csv\\dataset_2016.csv"  # Update with actual path
    sentences = load_data(filepath)
    vocab = build_vocab(sentences)
    data = numericalize_data(sentences, vocab)
    torch.save(vocab, "models/vocab.pth")  # Save vocabulary
    print("Data preprocessing complete.")
