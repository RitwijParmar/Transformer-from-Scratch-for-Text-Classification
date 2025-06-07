import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def load_and_preprocess_data(filepath):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(filepath)
    df = df[['text', 'airline_sentiment']]
    df['text'] = df['text'].str.lower()
    df.dropna(inplace=True)
    return df

def tokenize_texts(texts):
    """Tokenizes a list of texts using NLTK."""
    return [word_tokenize(str(text)) for text in texts]

def build_vocabulary(tokenized_texts):
    """Builds a vocabulary from tokenized texts."""
    vocab = build_vocab_from_iterator(tokenized_texts, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def numericalize_and_pad(tokenized_texts, vocab, max_len):
    """Converts tokens to integers and pads sequences."""
    numericalized = [torch.tensor([vocab[token] for token in text]) for text in tokenized_texts]
    padded = pad_sequence(numericalized, batch_first=True, padding_value=vocab["<pad>"])
    if padded.size(1) > max_len:
        padded = padded[:, :max_len]
    return padded

def create_dataloaders(df, batch_size=64, max_len=50):
    """Creates training, validation, and test dataloaders."""
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_encoded'] = df['airline_sentiment'].map(sentiment_map)

    tokenized_texts = tokenize_texts(df['text'].tolist())
    vocab = build_vocabulary(tokenized_texts)
    padded_texts = numericalize_and_pad(tokenized_texts, vocab, max_len)
    labels = torch.tensor(df['sentiment_encoded'].values)

    X_train, X_temp, y_train, y_temp = train_test_split(padded_texts, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab