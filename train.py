import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import StoryGenerator
from preprocess import load_data, build_vocab, numericalize_data

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.005

# Load dataset
filepath = "C:\\Users\\Shashank S M\\Desktop\\storyai\\dataset_2016.csv\\dataset_2016.csv"
sentences = load_data(filepath)
vocab = build_vocab(sentences)
data = numericalize_data(sentences, vocab)

# Convert to tensors
input_data = [torch.tensor(seq[:-1]) for seq in data if len(seq) > 1]
target_data = [torch.tensor(seq[1:]) for seq in data if len(seq) > 1]
dataset = TensorDataset(torch.nn.utils.rnn.pad_sequence(input_data, batch_first=True),
                        torch.nn.utils.rnn.pad_sequence(target_data, batch_first=True))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = StoryGenerator(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0), HIDDEN_DIM, NUM_LAYERS)
        outputs, _ = model(inputs, hidden)
        loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

torch.save(model.state_dict(), "models/story_generator_model.pth")
print("Training complete. Model saved.")
