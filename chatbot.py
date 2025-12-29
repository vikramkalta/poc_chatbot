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

from simple_evaluator import SimpleChatbotEvaluator


# Download required NLTK data
nltk.download("punkt_tab")


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)

        x = self.relu(self.layer2(x))
        x = self.dropout(x)

        x = self.relu(self.layer3(x))
        x = self.dropout(x)

        x = self.output_layer(x)
        return x


class SimpleChatbot:
    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
        self.words = []
        self.classes = []
        self.training_data = []

        # Load intents from JSON file
        intents_file = os.path.join(
            os.path.dirname(__file__), "remittance_intents.json"
        )
        # intents_file = os.path.join(os.path.dirname(__file__), 'intents.json')
        with open(intents_file, "r") as file:
            self.intents = json.load(file)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_data(self):
        """Prepare training data from intents"""
        # Reset any existing data
        self.words = []
        self.classes = []
        self.training_data = []

        print("Processing intents...")  # Debug print
        print(f"Number of intents: {len(self.intents['intents'])}")  # Debug print

        # First pass: collect all words and classes
        for intent in self.intents["intents"]:
            # Handle both formats: check if 'tag' or 'intent' is used
            tag = intent.get("tag", intent.get("intent", ""))
            if not tag:
                print(f"Warning: Intent missing 'tag' or 'intent' field: {intent}")
                continue

            self.classes.append(tag)
            patterns = intent.get("patterns", [])
            if not patterns:
                print(f"Warning: Intent '{tag}' has no patterns")
                continue

            for pattern in patterns:
                # Tokenize and stem each word
                words = nltk.word_tokenize(str(pattern))  # Ensure pattern is string
                self.words.extend([self.stemmer.stem(word.lower()) for word in words])

        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        print(f"Found {len(self.words)} unique words")  # Debug print
        print(f"Found {len(self.classes)} classes: {self.classes}")  # Debug print

        # Second pass: create training data
        X = []
        y = []

        for intent in self.intents["intents"]:
            tag = intent.get("tag", intent.get("intent", ""))
            if not tag or tag not in self.classes:
                continue

            patterns = intent.get("patterns", [])
            for pattern in patterns:
                # Bag of words for pattern
                bag = self._bag_of_words(str(pattern))  # Ensure pattern is string
                X.append(bag)

                # Create one-hot encoded output
                output_row = [0] * len(self.classes)
                output_row[self.classes.index(tag)] = 1
                y.append(output_row)

        if not X or not y:
            raise ValueError(
                "No valid training data found. Check your intents file format."
            )

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        print(f"Created {len(X)} training samples")  # Debug print
        print(f"Final X shape: {X.shape}, y shape: {y.shape}")  # Debug print
        return X, y

    def _bag_of_words(self, sentence):
        """Convert sentence to bag of words

        Args:
            sentence (str): Input sentence to convert to bag of words

        Returns:
            list: A bag of words representation of the input sentence
        """
        # Tokenize and stem the input sentence
        sentence_words = [
            self.stemmer.stem(word.lower()) for word in nltk.word_tokenize(sentence)
        ]

        # Initialize bag with 0 for each word
        bag = [0] * len(self.words)

        # Mark 1 for each word that exists in the sentence
        for i, word in enumerate(self.words):
            if word in sentence_words:
                bag[i] = 1

        return bag

    def train_model(self, epochs=1000, batch_size=16):
        """Train the neural network with batching for memory efficiency

        Args:
            epochs (int): Number of training epochs
            batch_size (int): Number of samples per batch
        """
        # Prepare training data
        X, y = self.preprocess_data()
        print(
            f"Raw X shape: {X.shape[0]} samples, {X.shape[1] if len(X.shape) > 1 else 1} features"
        )
        print(
            f"Raw y shape: {y.shape[0]} samples, {y.shape[1] if len(y.shape) > 1 else 1} classes"
        )

        # Convert to PyTorch datasets
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Initialize model
        input_size = X.shape[1]
        hidden_size = 8
        output_size = y.shape[1]

        self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()  # Better for multi-label classification
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        print(f"Training on {self.device}")
        print(
            f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}"
        )
        print(f"Batch size: {batch_size}, Total batches: {len(dataloader)}")

        # Training loop with batching
        for epoch in range(epochs):
            total_loss = 0.0
            self.model.train()  # Set model to training mode

            for batch_X, batch_y in dataloader:
                try:
                    # Move batch to device
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("Out of memory error. Try reducing batch size.")
                        torch.cuda.empty_cache()
                        return
                    else:
                        print(f"Error during training: {e}")
                        return

            # Print progress
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

        print("Training completed!")
        torch.cuda.empty_cache()  # Clean up CUDA cache

    def predict(self, sentence):
        """Predict intent for a given sentence"""
        if self.model is None:
            return "Model not trained yet!"

        # Preprocess input
        bag = self._bag_of_words(sentence)
        X = torch.FloatTensor(bag).to(self.device)
        X = X.unsqueeze(0)  # Add batch dimension

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            predicted = torch.argmax(output, dim=1)

        # Get intent tag
        tag = self.classes[predicted.item()]

        # Get random response
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

        return "Im not sure how respond to that."

    def test_predictions(self, test_phrases=None):
        """Test the model with some example phrases"""
        if test_phrases is None:
            test_phrases = [
                "How can I verify my account?",
                "What are your transfer fees?",
                "How long does a transfer take?",
                "Which countries do you support?",
                "What are your transfer limits?",
                "Hello!",
            ]

        print("\nTesting predictions:" + "=" * 50)
        for phrase in test_phrases:
            print(f"\nYou: {phrase}")
            response = self.predict(phrase)
            print(f"Chatbot: {response}")
        print("\n" + "=" * 60 + "\n")

    def chat(self, interactive=True):
        """Start interactive chat or run in non-interactive test mode

        Args:
            interactive (bool): If True, runs in interactive mode. If False, runs test predictions.
        """
        if not interactive:
            self.test_predictions()
            return

        print("Chatbot: Hello! I'm a simple FAQ chatbot. Type 'quit' to exit.")

        while True:
            try:
                user_input = input("You: ").strip().lower()

                if user_input in ["quit", "exit", "bye"]:
                    print("Chatbot: Goodbye!")
                    break

                response = self.predict(user_input)
                print(f"Chatbot: {response}")
            except EOFError:
                print("\nChatbot: Detected end of input. Switching to test mode...")
                self.test_predictions()
                break


# Create and train the chatbot
if __name__ == "__main__":
    chatbot = SimpleChatbot()

    # Train the model
    print("Training chatbot...")
    chatbot.train_model(epochs=1000)
    print("Training completed!")

    # Run in non-interactive mode by default for testing
    # To run interactively, call: chatbot.chat(interactive=True)
    chatbot.chat(interactive=False)

    # Step 2: Evaluation using trained components
    print("Step 2: Evaluation Using Model Components")
    # Create and run the evaluator
    evaluator = SimpleChatbotEvaluator(chatbot)
    results = evaluator.evaluate()

    # Test specific phrases
    evaluator.test_specific_phrases(
        [
            "How do I send money?",
            "What's the exchange rate?",
            "I need help with a transaction",
        ]
    )

    # Start chatting
    chatbot.chat()
