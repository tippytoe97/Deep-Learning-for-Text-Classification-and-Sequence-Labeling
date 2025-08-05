import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, vocab, num_classes):
        """
        Initialize the LSTM + Attention model with the embedding layer, bi-LSTM layer and a linear layer.
        
        Args:
            vocab: Vocabulary. (Refer to the documentation as specified in lstm.py)
            num_classes: Number of classes (labels).
        
        Returns:
            no returned value

        """
        super(Attention, self).__init__()

        self.embed_len = 50
        self.hidden_dim = 75
        self.n_layers = 1
        self.p = 0.5

        self.embedding_layer = nn.Embedding(len(vocab), self.embed_len, padding_idx=vocab['<pad>']) 
        self.lstm = nn.LSTM(input_size=self.embed_len,
        hidden_size = self.hidden_dim,
        num_layers = self.n_layers,
        batch_first=True,
        bidirectional= True,
        dropout = self.p if self.n_layers > 1 else 0) 
        self.linear = nn.Linear(self.hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(self.p) 
        self.context_layer = nn.Linear(self.hidden_dim*4, self.hidden_dim*2) 

    def forward(self, inputs, inputs_len):
        """
        Implement the forward function to feed the input through the model and get the output.

        1. Pass the input sequences through the embedding layer and lstm layer to obtain the lstm output and lstm final hidden state. This step should be implemented in forward_lstm().
        2. Compute the normalized attention weights from the lstm output and final hidden state. This step should be implemented in forward_attention().
        3. Compute the context vector, concatenate with the final hidden state and pass it through the context layer. This step should be implemented in forward_context().
        4. Pass the output from step 3 through the linear layer.

        Args:
            inputs : A (B, L) tensor containing the input sequences, where B = batch size and L = sequence length
            inputs_len :  A (B, ) tensor containing the lengths of the input sequences in the current batch prior to padding.

        Returns:
            output: Logits of each label. A tensor of size (B, C) where B = batch size and C = num_classes
        """
        lstm_out, final_hidden = self.forward_lstm(inputs, inputs_len)
        attention_weights = self.forward_attention(lstm_out, final_hidden)
        context = self.forward_context(lstm_out, attention_weights, final_hidden)

        context = self.dropout(context)
        output = self.linear(context)

        return output


    def forward_embed(self, inputs):
        return self.dropout(self.embedding_layer(inputs))

    def forward_lstm(self, inputs, inputs_len):
        """
        Pass the input sequences through the embedding layer and the LSTM layer to obtain the LSTM output and final hidden state.
        Concatenate the final forward and backward hidden states before returning.

        Args: 
            inputs : A (N, L) tensor containing the input sequences
            inputs_len : A (N, ) tensor containing the lengths of the input sequences prior to padding.

        Returns: 
            output : A (N, L', 2 * H) tensor containing the output of the LSTM. L' = the max sequence length in the batch (prior to padding) = max(inputs_len), and H = the hidden embedding size.
            hidden_concat : A (N, 2 * H) tensor containing the forward and backward hidden states concetenated along the last dimension.
        
        """
        embeddings = self.forward_embed(inputs)

        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, inputs_len.cpu(), batch_first = True, enforce_sorted= False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        forward_hidden = h_n[0]
        backward_hidden = h_n[1]
        hidden_concat = torch.cat((forward_hidden, backward_hidden), dim=1)

        return output, hidden_concat

    
    def forward_attention(self, lstm_output, hidden_concat):
        """
        Compute the unnormalized attention weights using the outputs of forward_lstm(). 
        Referenece: https://pytorch.org/docs/stable/generated/torch.bmm.html)
        Then, compute the normalized attention weights with the help of a softmax operation.

        Args:
            lstm_output : A (N, L', 2 * H) tensor containing the output of the LSTM.
            hidden_concat : A (N, 2 * H) tensor containing the forward and backward hidden states

        Returns:
            attention_weights : A (N, L') tensor containing the normalized attention weights.
        
        """

        attention_scores = torch.bmm(lstm_output, hidden_concat.unsqueeze(2))
        attention_scores = attention_scores.squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights

    
    def forward_context(self, lstm_output, attn_weights, hidden_concat):
        """
        Compute the context, which is the weighted sum of the lstm output (the coefficients are the attention weights). Then, concatenate
        the context with the hidden state, and pass it through the context layer + tanh().

        Args:
            lstm_output : A (N, L', 2 * H) tensor containing the output of the LSTM.
            attn_weights : A (N, L') tensor containing the normalized attention weights.
            hidden_concat : A (N, 2 * H) tensor containing the forward and backward hidden states

        Return:
            context_output : A (N, 2 * H) tensor containing the output of the context layer.


        """
        attn_weights = attn_weights.unsqueeze(1)
        attn_weights = attn_weights.float()

        context = torch.bmm(attn_weights, lstm_output)
        context = context.squeeze(1)
        combined = torch.cat((context, hidden_concat), dim=1)
        context_output = torch.tanh(self.context_layer(combined))

        return context_output













