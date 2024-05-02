import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

import json
import numpy as np
import pandas as pd
import seaborn as sns
from random import sample

def load_data(batch):
    with open('all_cards.json', 'r') as f:
        cards = json.load(f)
    card_ids = torch.load('card_ids.pt')[batch]
    attn_weights = [torch.load(f'attn_weights_l{layer}.pt')[batch] for layer in range(0, 2)]
    logits = torch.load('lm_logits.pt')
    selected_gt = torch.load('selected_labels.pt')
    return cards, card_ids, attn_weights, logits, selected_gt

def process_weights(attn_weights):
    # Mean across heads
    return [weights.mean(dim=0).cpu().detach() for weights in attn_weights]

def analyze_attention(card_ids, attn_weights, cards):
    results = []
    # Select 4 random tokens
    selected_tokens = sample(range(attn_weights[0].shape[-1]), 4)
    for token in selected_tokens:
        name = list(cards.keys())[int(card_ids[token].item())]
        token_data = {'Token': name}
        for i, layer_weights in enumerate(attn_weights):
            sorted_weights, sorted_indices = layer_weights[token].sort(descending=True)
            top_5 = [idx for idx in sorted_indices if idx != token][:5]
            bottom_2 = [idx for idx in sorted_indices if idx != token][-5:]
            token_data[f'Layer {i+1} Top'] = [(list(cards.keys())[int(card_ids[idx].item())], sorted_weights[idx].item()) for idx in top_5]
            token_data[f'Layer {i+1} Bottom'] = [(list(cards.keys())[int(card_ids[idx].item())], sorted_weights[idx].item()) for idx in bottom_2]
        results.append(token_data)
    return results

def visualize(results):
    num_layers = 2  # Assuming 4 layers as per your file loading
    num_cards = 10  # 3 top + 2 bottom cards per token

    # Create a large figure to accommodate all heatmaps
    fig, axs = plt.subplots(len(results), num_layers, figsize=(15, len(results) * 5), squeeze=False)

    for token_idx, result in enumerate(results):
        token_name = result['Token']
        all_layers_data = [result[f'Layer {i+1} Top'] + result[f'Layer {i+1} Bottom'] for i in range(num_layers)]

        for layer_idx, layer_data in enumerate(all_layers_data):
            names = [name for name, _ in layer_data]
            values = [weight for _, weight in layer_data]
            data = np.array(values).reshape(1, num_cards)  # Reshape for horizontal display

            # Plotting the heatmap
            sns.heatmap(data, ax=axs[token_idx, layer_idx], cmap='viridis', cbar=True, annot=True, fmt=".2f", 
                        xticklabels=names, yticklabels=[token_name])
            axs[token_idx, layer_idx].set_title(f'Layer {layer_idx+1}')
            axs[token_idx, layer_idx].tick_params(axis='x', rotation=45)
            axs[token_idx, layer_idx].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('attention_heatmap.png')
    plt.show()

def view_entire_attention(attention_weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(attention_weights.numpy(), cmap='viridis')

    # Adding a color bar to understand the scale of attention weights
    fig.colorbar(cax)

    # Setting titles and labels
    ax.set_title('Attention Weights Heatmap')
    ax.set_xlabel('Card Index Target')
    ax.set_ylabel('Card Index Source')

    # Displaying the plot
    plt.savefig('full_attention_heatmap.png')
    plt.show()

def main():
    batch = 3
    cards, card_ids, attn_weights, logits, selected_gt = load_data(batch)
    processed_weights = process_weights(attn_weights)
    # analysis_results = analyze_attention(card_ids, processed_weights, cards)
    # visualize(analysis_results)
    view_entire_attention(processed_weights[0])
    print(list(cards.keys())[logits[batch].argmax(-1)], list(cards.keys())[selected_gt[batch]])
    print([list(cards.keys())[i] for i in card_ids])
    print()

if __name__ == '__main__':
    main()
