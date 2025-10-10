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

    def create_test_dataset(self, test_size=0.2):
        """Create a test dataset from the training data"""
        if not self.chatbot.training_data:
            self.chatbot.prepare_training_data()
        
        # Shuffle and split the data
        random.shuffle(self.chatbot.training_data)
        split_idx = int(len(self.chatbot.training_data) * (1 - test_split))

        self.train_data = self.chatbot.training_data[:split_idx]
        self.test_data = self.chatbot.training_data[split_idx:]

        print(f"Training samples: {len(self.train_data)}")
        print(f"Testing samples: {len(self.test_data)}")

    def evaluate_accuracy(self):
        """Calculate accuracy on test set"""
        if not self.test_data:
            self.create_test_dataset()

        correct = 0
        total = len(self.test_data)
        all_predictions = []
        all_true_labels = []

        for test_input, true_label in self.test_data:
            # Convert bag of words back to text for prediction (simulation)
            # In real scenario, you'd have separate test patterns
            predicted_idx = self._predict_from_bag(test_input)
            true_idx = true_output.index(1) # Find which index has 1

            all_predictions.append(predicted_idx)
            all_true_labels.append(true_idx)

            if predicted_idx == true_idx:
                correct += 1
        
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f ({correct}/{total})}")

        return accuracy, all_predictions, all_true_labels
    
    def _predict_from_bag(self, bag):
        """Predict intent from bag of words directly"""
        X = torch.FloatTensor(bag).unsqueez(0).to(self.chatbot.device)

        self.chatbot.model.eval()
        with torch.no_grad():
            output = self.chatbot.model(X)
            predicted = torch.argmax(output, dim=1)

        return predicted.item()

    def confusion_matrix_analysis(self, predictions, true_labels):
        """Create and display confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.chatbot.classes,
                   yticklabels=self.chatbot.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def classification_report(self, predictions, true_labels):
        """Generate detailed classification report"""
        report = classification_report(true_labels, predictions, 
                                     target_names=self.chatbot.classes,
                                     output_dict=True)
        
        # Print formatted report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(true_labels, predictions, 
                                  target_names=self.chatbot.classes))
        
        return report
    
    def per_class_accuracy(self, predictions, true_labels):
        """Calculate accuracy for each intent class"""
        class_correct = {cls: 0 for cls in self.chatbot.classes}
        class_total = {cls: 0 for cls in self.chatbot.classes}
        
        for pred_idx, true_idx in zip(predictions, true_labels):
            true_class = self.chatbot.classes[true_idx]
            class_total[true_class] += 1
            
            if pred_idx == true_idx:
                class_correct[true_class] += 1
        
        print("\n" + "="*50)
        print("PER-CLASS ACCURACY")
        print("="*50)
        for cls in self.chatbot.classes:
            if class_total[cls] > 0:
                acc = class_correct[cls] / class_total[cls]
                print(f"{cls:15}: {acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
        
        return class_correct, class_total
            
    def confidence_analysis(self, num_samples=50):
        """Analyze prediction confidence scores"""
        if not self.test_data:
            self.create_test_dataset()

        confidences = []
        correct_confidences = []
        incorrect_confidences = []

        for i, (test_input, true_output) in enumerate(self.test_data[:num_samples]):
            X = torch.FloatTensor(test_input).unsqueeze(0).to(self.chatbot.device)

            self.chatbot.model.eval()
            with torch.no_grad():
                output = self.chatbot.model(X)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)

            true_idx = true_output.index(1)
            confidence_value = confidence.item()
            confidences.append(confidence_value)

            if predicted.item() == true_idx:
                correct_confidences.append(confidence_value)
            else:
                incorrect_confidences.append(confidence_value)

        print(f"\nAverage confidence: {np.mean(confidences):.4f}")
        print(f"Correct predictions confidence: {np.mean(correct_confidences) if correct_confidences else 0:.4f}")
        print(f"Incorrect predictions confidence: {np.mean(incorrect_confidences) if incorrect_confidences else 0:.4f}")
        
        return confidences, correct_confidences, incorrect_confidences

    def run_complete_evaluation(self):
        """Run all evaluation metrics"""
        print("🚀 RUNNING COMPLETE MODEL EVALUATION")
        print("="*60)
        
        # 1. Basic accuracy
        accuracy, predictions, true_labels = self.evaluate_accuracy()
        
        # 2. Per-class accuracy
        self.per_class_accuracy(predictions, true_labels)
        
        # 3. Classification report
        self.classification_report(predictions, true_labels)
        
        # 4. Confusion matrix
        self.confusion_matrix_analysis(predictions, true_labels)
        
        # 5. Confidence analysis
        self.confidence_analysis()
        
        return accuracy