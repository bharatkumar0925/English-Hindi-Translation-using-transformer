import time
start = time.time()
from translation_metrics import evaluate_metrics
from architecture import Encoder, Decoder, Seq2Seq
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from clean_text import clean_text

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the dataset
path = r"C:\Users\BHARAT\Desktop\data sets\text datasets\english hindi words\Dataset_English_Hindi.csv"
lines = pd.read_csv(path, nrows=10000).dropna()
lines.columns = lines.columns.str.lower()
english_words, hindi_words = clean_text(lines, 'english', 'hindi')

# Create token indices
input_token_index = {word: i + 1 for i, word in enumerate(sorted(english_words))}
target_token_index = {word: i + 1 for i, word in enumerate(sorted(hindi_words))}

# Prepare training and testing data
X, y = lines['english'], lines['hindi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)

max_length_src = max(lines['english'].str.split().apply(len))
max_length_tar = max(lines['hindi'].str.split().apply(len))

def encode_sequences(token_index, sequences, max_len):
    encoded = []
    for sequence in sequences:
        encoded_seq = [token_index.get(word, 0) for word in sequence.split()]
        padded_seq = encoded_seq+[0]*(max_len-len(encoded_seq))
        encoded.append(padded_seq)
    return np.array(encoded)


X_train_encoded = encode_sequences(input_token_index, X_train, max_length_src)
y_train_encoded = encode_sequences(target_token_index, y_train, max_length_tar)
X_test_encoded = encode_sequences(input_token_index, X_test, max_length_src)
y_test_encoded = encode_sequences(target_token_index, y_test, max_length_tar)


X_train_tensor = torch.tensor(X_train_encoded, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_encoded, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)

class TranslationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TranslationDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TranslationDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

input_dim = len(input_token_index)+1
output_dim = len(target_token_index)+1
embed_dim = 1024
hidden_dim = 1024
num_layers = 2
encoder = Encoder(input_dim, embed_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trz) in enumerate(iterator):
        optimizer.zero_grad()
        output = model(src, trz)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trz = trz[:, 1:].reshape(-1)
        loss = criterion(output, trz)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        epoch_loss+=loss.item()
    return epoch_loss/len(iterator)


n_epochs = 10
clip = 1

for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, clip)
    test_loss = train(model, test_loader, optimizer, criterion, clip)
    print(f'Epoch{epoch+1}/{n_epochs}, loss: {train_loss:.4f}\nTest loss: {test_loss:.4f}')

torch.save(model, 'english-hindi-model.pth')
def translate_sentence(sentence, model, input_token_index, target_token_index, max_len):
    model.eval()
    tokens = [input_token_index.get(word, 0) for word in sentence.split()]
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(tokens)

    trz_indices = [target_token_index['START_']]
    for _ in range(max_len):
        trz_tensor = torch.LongTensor([trz_indices[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trz_tensor, hidden, cell, encoder_outputs)

        pred_token = output.argmax(1).item()
        trz_indices.append(pred_token)

        if pred_token == target_token_index['_END']:
            break

    # Convert token indices back to words
    trz_tokens = [list(target_token_index.keys())[list(target_token_index.values()).index(i)] for i in trz_indices]

    # Remove the START and END tokens from the output
    return trz_tokens[1:-1]


sentence = 'how are you'
translated_token = translate_sentence(sentence, model, input_token_index, target_token_index, max_length_tar)


translated_sentence = ' '.join(translated_token)
print(translated_sentence)


y_true = []  # List to hold ground truth (y_test)
y_pred = []  # List to hold predicted sentences

# Iterate over your test data
for sentence, true_translation in zip(X_test, y_test):
    # Translate the sentence
    predicted_tokens = translate_sentence(sentence, model, input_token_index, target_token_index, max_len=50)
    predicted_sentence = ' '.join(predicted_tokens)

    # Add to the lists
    y_true.append(true_translation)
    y_pred.append(predicted_sentence)

# Now compute BLEU scores


evaluate_metrics(y_true, y_pred)


end = time.time()
print(f'Total time: {end-start}')
