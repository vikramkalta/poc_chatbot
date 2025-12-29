import json
import random
from typing import List, Dict, Tuple

class SimpleChatbotEvaluator:
    def __init__(self, chatbot):
        """
        Initialize the evaluator with a chatbot instance.
        
        Args:
            chatbot: An instance of your SimpleChatbot class
        """
        self.chatbot = chatbot
        self.test_samples = []
    
    def create_test_samples(self, test_split: float = 0.2) -> None:
        """
        Create test samples from the intents data.
        
        Args:
            test_split: Fraction of data to use for testing (0.0 to 1.0)
        """
        self.test_samples = []
        
        # For each intent, take some patterns as test samples
        for intent in self.chatbot.intents['intents']:
            tag = intent['tag']
            patterns = intent['patterns']
            
            # Calculate how many patterns to use for testing (at least 1)
            num_test = max(1, int(len(patterns) * test_split))
            test_patterns = random.sample(patterns, num_test)
            
            # Add to test samples
            for pattern in test_patterns:
                self.test_samples.append({
                    'text': pattern,
                    'expected_tag': tag
                })
        
        print(f"Created {len(self.test_samples)} test samples")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the chatbot on test samples.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.test_samples:
            self.create_test_samples()
        
        correct = 0
        results = []
        
        for sample in self.test_samples:
            # Get the chatbot's prediction
            response = self.chatbot.predict(sample['text'])
            
            # Since predict only returns the response text, we need to get the predicted tag
            # We'll find which intent's response matches the prediction
            predicted_tag = None
            confidence = 0.0  # Placeholder since we don't have confidence from predict()
            
            # Find which intent's response matches the prediction
            for intent in self.chatbot.intents['intents']:
                if response in intent['responses']:
                    predicted_tag = intent['tag']
                    break
            
            # If no matching response found, use the first tag (this is a fallback)
            if predicted_tag is None and self.chatbot.classes:
                predicted_tag = self.chatbot.classes[0]
            
            # Check if prediction is correct
            is_correct = (predicted_tag == sample['expected_tag'])
            if is_correct:
                correct += 1
            
            # Store results for analysis
            results.append({
                'text': sample['text'],
                'expected': sample['expected_tag'],
                'predicted': predicted_tag,
                'confidence': confidence,
                'correct': is_correct,
                'response': response
            })
        
        # Calculate metrics
        accuracy = correct / len(self.test_samples)
        
        # Print summary
        print("\n" + "="*50)
        print(f"EVALUATION RESULTS")
        print("="*50)
        print(f"Total test samples: {len(self.test_samples)}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f} ({correct}/{len(self.test_samples)})")
        
        # Print some examples of incorrect predictions
        incorrect = [r for r in results if not r['correct']]
        if incorrect:
            print("\n" + "-"*50)
            print("EXAMPLES OF INCORRECT PREDICTIONS")
            print("-"*50)
            for i, result in enumerate(incorrect[:5]):  # Show up to 5 examples
                print(f"\nExample {i+1}:")
                print(f"Text: {result['text']}")
                print(f"Expected: {result['expected']}")
                print(f"Predicted: {result['predicted']} (Confidence: {result['confidence']:.2f})")
        
        return {
            'total_samples': len(self.test_samples),
            'correct_predictions': correct,
            'accuracy': accuracy,
            'incorrect_examples': [r for r in results if not r['correct']]
        }
    
    def test_specific_phrases(self, phrases: List[str]):
        """
        Test specific phrases and print the results.
        
        Args:
            phrases: List of phrases to test
        """
        print("\n" + "="*50)
        print("TESTING SPECIFIC PHRASES")
        print("="*50)
        
        for phrase in phrases:
            response = self.chatbot.predict(phrase)
            
            # Find which intent's response matches the prediction
            predicted_tag = None
            for intent in self.chatbot.intents['intents']:
                if response in intent['responses']:
                    predicted_tag = intent['tag']
                    break
            
            print(f"\nInput: {phrase}")
            print(f"Predicted intent: {predicted_tag or 'Unknown'}")
            print(f"Response: {response}")

# Example usage
if __name__ == "__main__":
    from chatbot import SimpleChatbot
    
    # Initialize and train the chatbot
    chatbot = SimpleChatbot()
    chatbot.train_model()  # Make sure to train the model first
    
    # Create and run the evaluator
    evaluator = SimpleChatbotEvaluator(chatbot)
    
    # Option 1: Evaluate on test samples from intents
    print("Running evaluation on test samples...")
    results = evaluator.evaluate()
    
    # Option 2: Test specific phrases
    test_phrases = [
        "How do I send money?",
        "What's the exchange rate?",
        "I need help with a transaction"
    ]
    evaluator.test_specific_phrases(test_phrases)
