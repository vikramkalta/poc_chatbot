import numpy as np
import random
import json
import os
from numpy.random import rand
import nltk
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
import torch.optim as optim

# Download required NLTK data
nltk.download('punkt_tab')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class SimpleChatbot:
    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
        self.words = []
        self.classes = []
        self.training_data = []

        # Load intents from JSON file
        intents_file = os.path.join(os.path.dirname(__file__), 'remittance_intents.json')
        with open(intents_file, 'r') as file:
            self.intents = json.load(file)

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_data(self):
        """Prepare training data from intents"""
        # Extract words and classes
        for intent in self.intents['intents']:
            self.classes.append(intent['tag'])
            for pattern in intent['patterns']:
                # Tokenize and stem each word
                words = nltk.word_tokenize(pattern)
                self.words.extend([self.stemmer.stem(word.lower()) for word in words])

        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        # Create training data
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Bag of words for pattern
                bag = self._bag_of_words(pattern)

                # Out row (one-hot encoded)
                output_row = [0] * len(self.classes)
                output_row[self.classes.index(intent['tag'])] = 1

                self.training_data.append([bag, output_row])

    def _bag_of_words(self, sentence):
        """Convert sentence to bag of words"""
        # Tokenize and stem
        sentence_words = [self.stemmer.stem(word.lower()) for word in nltk.word_tokenize(sentence)]

        # Initialise bag with 0 for each word
        bag = [0] * len(self.words)

        # Set to 1 if word exists in sentence
        for i, word in enumerate(self.words):
            if word in sentence_words:
                bag[i] = 1
        
        return bag

    def train_model(self, epochs=1000):
        """Train the neural network"""
        self.preprocess_data()

        # Convert to numpy arrays
        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])

        # Convert to PyTorch tensors
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        # Initialise model
        input_size = len(X[0])
        hidden_size = 8
        output_size = len(y[0])

        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, sentence):
        """Predict intent for a given sentence"""
        if self.model is None:
            return "Model not trained yet!"

        # Preprocess input
        bag = self._bag_of_words(sentence)
        X = torch.FloatTensor(bag).to(self.device)
        X = X.unsqueeze(0) # Add batch dimension
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            predicted = torch.argmax(output, dim=1)
        
        # Get intent tag
        tag = self.classes[predicted.item()]

        # Get random response
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        
        return "Im not sure how respond to that."

    
    def chat(self):
        """Start interactive chat"""
        print("Chatbot: Hello! I'm a simple FAQ chatbot. Type 'quit' to exit.")

        while True:
            user_input = input("You: ").strip().lower()

            if user_input in ['quit', 'exit', 'bye']:
                print("Chatbot: Goodbye!")
                break

            if user_input:
                response = self.predict(user_input)
                print(f"Chatbot: {response}")

# Create and train the chatbot
if __name__ == "__main__":
    chatbot = SimpleChatbot()

    print("Training chatbot...")
    chatbot.train_model(epochs=1000)
    print("Training completed!")

    # Start chatting
    chatbot.chat()