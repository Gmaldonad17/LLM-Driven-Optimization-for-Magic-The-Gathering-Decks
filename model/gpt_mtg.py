from transformers import GPTNeoForCausalLM
import torch.nn as nn

class GPTNeoForMTG(GPTNeoForCausalLM):
    def __init__(self, config, 
                 iterations, 
                 cards,
                 ):
        super().__init__(config)
        self.iterations = iterations
        self.cards = cards
        self.lm_head = nn.Linear(self.lm_head.in_features, len(self.cards), bias=False)
        # Add any additional initialization or modifications here

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Modify the forward function to suit your needs
        
        outputs = super().forward(input_ids, attention_mask, **kwargs)
        
        # Perform any additional operations or modifications to the outputs
        
        return outputs