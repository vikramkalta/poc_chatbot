from cgi import test
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class ChatbotEvaluator:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.test_data = []
        self.predictions = []
        self.true_labels = []

    def create_test_dataset(self, test_size=0.2):
        """Split data into train/test sets"""
        all_patterns = []
        all_labels = []

        for intent in self.chatbot.intents['intents']:
            for pattern in intent['patterns']:
                all_patterns.append(pattern)
                all_labels.append(intent['tag'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            all_patterns, all_labels, test_size=test_size, random_state=42, stratify=all_labels
        )

        self.test_data = list(zip(X_test, y_test))
        return X_train, X_test, y_train, y_test

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        if not self.test_data:
            self.create_test_dataset()

        self.predictions = []
        self.true_labels = []

        print("Running Comprehensive Evaluation...")
        print("=" * 50)

        # Test each example
        for text, true_label in self.test_data:
            predicted_intent, confidence = self.chatbot.predicted_intent(text)
            self.true_labels.append(true_label)

        # Calculate metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        
            