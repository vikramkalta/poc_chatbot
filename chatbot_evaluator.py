from cgi import test
import numpy as np
from seaborn.matrix import heatmap
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
            self.predictions.append(predicted_intent)
            self.true_labels.append(true_label)

        # Calculate metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        
        # Print results
        print(f"Model Performance Metrics:")
        print(f" Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Test Size: {len(self.test_data)} samples")

        # Confusion matrix
        self.plot_confusion_matrix()
        
        # Detailed class-wise performance
        self.print_class_metrics()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }  

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.true_labels, self.predictions, labels=self.chatbot.classes)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.chatbot.classes,
                yticklabels=self.chatbot.classes)   
        plt.figure('Confusion matrix - chatbot performance')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()    

    def print_class_metrics(self):
        """Print performance for each intent class"""
        print(f"Class-wise performance")   
        print("-" * 40)

        for class_name in self.chatbot.classes:
            # Calculate metrics for this class
            true_positives = sum(1 for true, pred in zip(self.true_labels, self.predictions) 
            if true == class_name and pred == class_name)
            actual_positives = sum(1 for true in self.true_labels if true == class_name)
            predicted_positives = sum(1 for pred in self.predictions if pred == class_name)
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f" {class_name:20} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f} | Samples: {actual_positives}")

    def test_specific_queries(self, test_queries):
        """Test specific user queries"""
        print(f"Testing specific queries:")
        print("-" * 40)

        for query in test_queries:
            intent, confidence = self.chatbot.predict_intent(query)
            response = self.chatbot.get_response(intent)
            print(f" Query: '{query}'")
            print(f" -> Intent: {intent} (Confidence: {confidence:.3f})")
            print(f" -> Response: {response}")

    def calculate_confidence_distribution(self):
        """Analyze confidence scores"""
        print(f"Confidence distribution analysis:")
        print("-" * 40)

        confidences = []
        correct_confidences = []
        incorrect_confidences = []

        for (text, true_label), pred in zip(self.test_data, self.predictions):
            intent, confidence = self.chatbot.predict_intent(text)
            confidences.append(confidence)

            if pred == true_label:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)

        print(f"   Average Confidence: {np.mean(confidences):.3f}")
        print(f"   Correct Predictions Avg Confidence: {np.mean(correct_confidences):.3f}")
        print(f"   Incorrect Predictions Avg Confidence: {np.mean(incorrect_confidences):.3f}")
        print(f"   Confidence Std Dev: {np.std(confidences):.3f}")

# Enhanced Chatbot with Performance Tracking
class EnhancedRemittanceChatbot(EnhanchedChatbot):
    def __init__(self, intents_data=None):
        if intents_data is None:
            intents_data = remittance_intents
        super().__init__()
        self.intents = intents_data
        self.performance_history = []

    def train_with_validation(self, epochs=1000):
        """Train with performance tracking"""
        print("Training Enhanced Remittance Chatbot...")
        self.prepare_training_data()

        # Split data
        evaluator = ChatbotEvaluator(self)
        X_train, X_test, y_train, y_test = evaluator.create_test_dataset()

        # Convert training data format
        train_data = []
        for text, label in zip(X_train, y_train):
            bag = self._bag_of_words(text)
            output_row = [0] * len(self.classes)
            output_row[self.classes.index(label)] = 1
            train_data.append([bag, output_row])
        
        # Training (simplified - you'd adapt your actual training here)
        self.training_data = train_data
        self.train(epochs=epochs)

        # Evaluate
        print("\n" + "=" * 50)
        print("Final performance analysis")
        print("="*60)
        
        metrics = evaluator.evaluate_model()
        self.performance_history.append(metrics)
        
        # Test specific scenarios
        test_queries = [
            "how much does it cost to send money to india?",
            "what is your customer support number?",
            "can I cancel my transfer?",
            "how long will the money take to arrive?",
            "which countries do you support?"
        ]

        evaluator.test_specific_queries(test_queries)
        evaluator.calculate_confidence_distribution()
        
        return metrics

# Usage Example
if __name__ == "__main__":
    # Create and train the enhanced chatbot
    chatbot = EnhancedRemittanceChatbot(remittance_intents)
    # Train and evaluate
    performance = chatbot.train_with_validation(epochs=800)
    print(f"\n✅ Training completed! Key metric - Accuracy: {performance['accuracy']*100:.1f}%")
    
    # Interactive chat
    print("\n💬 Starting interactive chat session...")
    chatbot.chat()