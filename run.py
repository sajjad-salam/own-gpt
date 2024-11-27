import torch
from torch import nn
import re

# Define the SimpleLSTM model


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

# Load and preprocess data


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

# Load the model with the correct architecture


def load_model(model_path, vocab_size, embed_size=128, hidden_size=256):
    model = SimpleLSTM(vocab_size, embed_size, hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Generate text using the model


def generate_text(model, start_text, word2idx, idx2word, max_length=20):
    words = start_text.lower().split()
    for _ in range(max_length):
        x = torch.tensor([[word2idx.get(word, 0)
                         for word in words[-5:]]], dtype=torch.long)
        output = model(x)
        _, predicted = torch.max(output, dim=1)
        next_word = idx2word.get(predicted.item(), "")
        words.append(next_word)
    return ' '.join(words)


# Load data and prepare vocabulary
data = load_data("data.txt")
word2idx, idx2word = create_vocabulary(data)

vocab_size = len(word2idx)

# Ensure the model architecture matches the saved model
model = load_model("simple_lstm_model.pth", vocab_size,
                   embed_size=128, hidden_size=256)

# Start the chatbot
print("Chatbot: Hello! I'm ready to chat with you. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = generate_text(
        model, user_input, word2idx, idx2word, max_length=20)
    print("Chatbot:", response)
