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
        X = X.unsqueeze(0) # Add batch dimension

        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks for each layer
        hooks = []
        hooks.append(self.model.layer1.register_forward_hook(get_activation('layer1')))
        hooks.append(self.model.layer2.register_forward_hook(get_activation('layer2')))
        hooks.append(self.model.layer3.register_forward_hook(get_activation('layer3')))
        hooks.append(self.model.output_layer.register_forward_hook(get_activation('output_layer')))

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
        print("="*60)
        print("NEURON ACTIVATION ANALYSIS")
        print("="*60)

        all_activations = {}

        for i, sentence in enumerate(sentences):
            print(f"\nSentence {i+1}: '{sentence}'")
            activations = self.get_layer_activations(sentence)

            if activations:
                all_activations[sentence] = activations

                # Analyse each layer
                for layer_name in ['layer1', 'layer2', 'layer3']:
                    layer_act = activations[layer_name][0] # Remove batch dimension

                    # Get top activating neuron
                    top_indices = np.argsort(layer_act)[-top_k:]
                    top_activations = layer_act[top_indices]

                    print(f" {layer_name} - Top {top_k} neurons:")
                    for idx, act_val in zip(top_indices, top_activations):
                        print(f"    Neuron {idx}: {act_val:.4f}")

        return all_activations
