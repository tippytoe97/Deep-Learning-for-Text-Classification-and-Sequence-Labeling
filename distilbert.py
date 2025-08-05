import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import DistilBertModel


class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the following modules:
            1. DistilBert Model using the pretrained 'distilbert-base-uncased' model
            2. Linear layer
            3. Any other layers to help with accuracy

        Args:
            num_classes: Number of classes (labels).

        """
        super(DistilBERTClassifier, self).__init__()
        # Load pretrained DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.3)
        # Linear layer to map DistilBERT hidden states to number of classes
        self.linear = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, inputs, mask):
        """
        Implement the forward function to feed the input through the distilbert model with inputs and mask.
        Use the DistilBert output to obtain logits of each label. 
        Ref: https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel

        Args:
            inputs: Input data. (B, L) tensor of tokens where B is batch size and L is max sequence length.
            mask: attention_mask. (B, L) tensor of binary mask.

        Returns:
            output: Logits of each label. (B, C) tensor of logits where C is number of classes.
        """

        # Feed inputs and mask into DistilBERT
        distilbert_output = self.distilbert(input_ids=inputs, attention_mask=mask)
        # distilbert_output.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        # Use the first token's embedding ([CLS] equivalent) for classification:
        hidden_state = distilbert_output.last_hidden_state  # (B, L, H)
        pooled_output = hidden_state[:, 0]  # Take embedding of [CLS] token (first token)

        dropped = self.dropout(pooled_output)  # Apply dropout
        logits = self.linear(dropped)          # Linear layer to get logits
        return logits
