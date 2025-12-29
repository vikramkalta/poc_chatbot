from curses import noecho
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np


class NeuralNetAnalyzer:
    def __init__(self, chatbot):
        """
        Initialise analyzer with access to chatbot and its model

        Args:
            chatbot: Your SimpleChatbot instance
        """
        self.chatbot = chatbot
        self.model = chatbot.model
        self.device = chatbot.device

    def get_layer_activations(self, sentence):
        """
        Get activations for each layer for a given sentence
        """
        if self.model is None:
            print("Model not trained yet")
            return None

        # Preproces input
        bag = self.chatbot._bag_of_words(sentence)
        X = torch.FloatTensor(bag).to(self.device)
        X = X.unsqueeze(0)  # Add batch dimension

        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu().numpy()

            return hook

        # Register hooks for each layer
        hooks = []
        hooks.append(self.model.layer1.register_forward_hook(get_activation("layer1")))
        hooks.append(self.model.layer2.register_forward_hook(get_activation("layer2")))
        hooks.append(self.model.layer3.register_forward_hook(get_activation("layer3")))
        hooks.append(
            self.model.output_layer.register_forward_hook(
                get_activation("output_layer")
            )
        )

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(X)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    def visualize_neuron_activations(self, sentences, top_k=10):
        """
        Visualise which neurons activate most for different sentences
        """
        print("=" * 60)
        print("NEURON ACTIVATION ANALYSIS")
        print("=" * 60)

        all_activations = {}

        for i, sentence in enumerate(sentences):
            print(f"\nSentence {i+1}: '{sentence}'")
            activations = self.get_layer_activations(sentence)

            if activations:
                all_activations[sentence] = activations

                # Analyse each layer
                for layer_name in ["layer1", "layer2", "layer3"]:
                    layer_act = activations[layer_name][0]  # Remove batch dimension

                    # Get top activating neuron
                    top_indices = np.argsort(layer_act)[-top_k:][::-1]
                    top_activations = layer_act[top_indices]

                    print(f" {layer_name} - Top {top_k} neurons:")
                    for idx, act_val in zip(top_indices, top_activations):
                        print(f"    Neuron {idx}: {act_val:.4f}")

        return all_activations

    def inspect_layer_weights(self, top_words_per_neuron=5):
        """
        Inspect which input words are most important for each neuron
        """
        if self.model is None or not hasattr(self.chatbot, "words"):
            print("Model or vocabulary not available")
            return

        print("\n" + "=" * 60)
        print("LAYER WEIGHT INSPECTION")
        print("=" * 60)

        # Layer 1 weights (most interpretable)
        weights_layer1 = (
            self.model.layer1.weight.data.cpu().numpy()
        )  # Shape: (128, input_size)

        print(
            f"\nLAYER 1 WEIGHTS (128 neurons, {weights_layer1.shape[1]} input features)"
        )
        print("-" * 50)

        # Analyze each neuron in layer 1
        for neuron_idx in range(
            min(20, weights_layer1.shape[0])
        ):  # Show first 20 neurons
            neuron_weights = weights_layer1[neuron_idx]

            # Get most positive and negative weights
            top_positive_indices = np.argsort(neuron_weights)[-top_words_per_neuron:][
                ::-1
            ]
            top_negative_indices = np.argsort(neuron_weights)[:top_words_per_neuron]

            print(f"\nNeuron {neuron_idx}")

            # Positive weights (words that activate this neuron)
            print("  Activates for:")
            for word_idx in top_positive_indices:
                if word_idx < len(self.chatbot.words):
                    weight = neuron_weights[word_idx]
                    print(f"  '{self.chatbot.words[word_idx]}': {weight:.4f}")

            # Negative weights (words that suppress this neuron)
            print(" Suppressed by:")
            for word_idx in top_negative_indices:
                if word_idx < len(self.chatbot.words):
                    weight = neuron_weights[word_idx]
                    print(f" '{self.chatbot.words[word_idx]}': {weight:.4f}")

    def visualize_weight_distributions(self):
        """
        Plot weight distributions for each layer
        """
        if self.model is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        layers = [
            ("Layer 1", self.model.layer1.weight.data.cpu().numpy().flatten()),
            ("Layer 2", self.model.layer2.weight.data.cpu().numpy().flatten()),
            ("Layer 3", self.model.layer3.weight.data.cpu().numpy().flatten()),
            (
                "Output Layer",
                self.model.output_layer.weight.data.cpu().numpy().flatten(),
            ),
        ]

        for i, (layer_name, weights) in enumerate(layers):
            axes[i].hist(
                weights, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
            )
            axes[i].set_title(f"{layer_name} Weight Distribution")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

            # Add statistics
            mean = np.mean(weights)
            std = np.std(weights)
            axes[i].axvline(
                mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}"
            )
            axes[i].axvline(
                mean + std, color="orange", linestyle="--", alpha=0.7, label=f"±1 std"
            )
            axes[i].axvline(mean - std, color="orange", linestyle="--", alpha=0.7)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_neuron_activations_heatmap(self, sentences):
        """
        Create heatmap of neuron activatons across sentences
        """
        activations_data = []
        sentence_labels = []

        for sentence in sentences:
            acts = self.get_layer_activations(sentence)
            if acts:
                # Use layer 1 activations
                layer1_acts = acts["layer1"][0]
                activations_data.append(layer1_acts)
                sentence_labels.append(
                    sentence[:30] + "..." if len(sentence) > 30 else sentence
                )

        if not activations_data:
            return

        activations_matrix = np.array(activations_data)

        plt.figure(figsize=(15, 8))
        sns.heatmap(
            activations_matrix,
            xticklabels=[f"Neuron {i}" for i in range(activations_matrix.shape[1])],
            yticklabels=sentence_labels,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Activation Value"},
        )
        plt.title("Neuron Activations Heatmap (Layer 1)")
        plt.xlabel("Neuron Index")
        plt.ylabel("Input Sentences")
        plt.tight_layout()
        plt.show()

    def analyze_output_layer(self):
        """
        Analyze which features lead to which output classes
        """
        if self.model is None or not hasattr(self.chatbot, "classes"):
            return

        output_weights = self.model.output_layer.weight.data.cpu().numpy()  # (15, 32)

        print("\n" + "=" * 60)
        print("OUTPUT LAYER ANALYSIS")
        print("=" * 60)

        for class_idx, class_name in enumerate(self.chatbot.classes):
            class_weights = output_weights[class_idx]

            # Get most important features for this class
            top_feature_indices = np.argsort(np.abs(class_weights))[-10:][::-1]

            print(f"\n{class_name}:")
            print("  Most important features from Layer 3:")
            for feat_idx in top_feature_indices:
                weight = class_weights[feat_idx]
                print(f"    Feature {feat_idx}: {weight:.4f}")

    def comprehensive_analysis(self, test_sentences=None):
        """
        Run all analysis methods
        """
        if test_sentences is None:
            test_sentences = [
                "How do I send money?",
                "What's the exchange rate for USD?",
                "I need help with my transaction",
                "What are your fees?",
                "How long does transfer take?",
                "Can I cancel my payment?",
            ]

        print("COMPREHENSIVE NEURAL NETWORK ANALYSIS")
        print("=" * 60)

        # 1. Neuron activations
        activations = self.visualize_neuron_activations(test_sentences, top_k=8)

        # 2. Weight inspection
        self.inspect_layer_weights(top_words_per_neuron=3)

        # 3. Output layer analysis
        self.analyze_output_layer()

        # 4. Visualisations
        self.visualize_weight_distributions()
        self.plot_neuron_activations_heatmap(test_sentences)

        return activations


if __name__ == "__main__":
    from chatbot import SimpleChatbot

    # Initialise and train the chatbot
    chatbot = SimpleChatbot()
    chatbot.train_model()

    # Create analyzer
    analyzer = NeuralNetAnalyzer(chatbot)

    # Run comprehensive analysis
    test_sentences = [
        "How do I send money?",
        "What's the exchange rate?",
        "I need help with a transaction",
        "What are your fees?",
        "How long does transfer take?",
        "Can I cancel my payment?",
    ]

    analyzer.comprehensive_analysis(test_sentences)

    # Also run your existing evaluator
    evaluator = SimpleChatbotEvaluator(chatbot)
    results = evaluator.evaluate()
