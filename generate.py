import torch
from model import StoryGenerator
from preprocess import tokenizer, build_vocab

# Load trained model
vocab = torch.load("models/vocab.pth")
model = StoryGenerator(len(vocab), 128, 256, 2)
model.load_state_dict(torch.load("models/story_generator_model.pth"))
model.eval()

def generate_story(start_word, max_len=50):
    words = [start_word]
    hidden = model.init_hidden(1, 256, 2)
    input_tensor = torch.tensor([[vocab[start_word]]])

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            predicted_token = torch.argmax(output[:, -1, :], dim=-1).item()
            next_word = vocab.lookup_token(predicted_token)
            words.append(next_word)
            if next_word == "<pad>":
                break
            input_tensor = torch.tensor([[predicted_token]])

    return " ".join(words)

if __name__ == "__main__":
    start_word = input("Enter a start word: ")
    print(generate_story(start_word))
