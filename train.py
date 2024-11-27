import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re


def load_data(filename="data.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.split()


def create_vocabulary(data):
    vocab = sorted(set(data))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


class TextDataset(Dataset):
    def __init__(self, data, word2idx, sequence_length=5):
        self.data = data
        self.word2idx = word2idx
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length]
        y = self.data[index + self.sequence_length]
        x = torch.tensor([self.word2idx[word] for word in x], dtype=torch.long)
        y = torch.tensor(self.word2idx[y], dtype=torch.long)
        return x, y


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


data = load_data("data.txt")
word2idx, idx2word = create_vocabulary(data)
dataset = TextDataset(data, word2idx)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

vocab_size = len(word2idx)
embed_size = 128
hidden_size = 256
model = SimpleLSTM(vocab_size, embed_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "simple_lstm_model.pth")
print("Model training completed and saved.")
