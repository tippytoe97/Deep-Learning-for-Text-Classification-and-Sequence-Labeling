import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab, num_classes):
        """
        Initialize LSTM with the embedding layer, LSTM layer and a linear layer.
        
        Args:
            vocab: Vocabulary. (Refer to this for documentation: https://pytorch.org/text/stable/vocab.html)
            num_classes: Number of classes (labels).

        Returns:
            no returned value

        NOTE: Use the following variable names to initialize the parameters:
            1. self.embed_len -> the embedding dimension
            2. self.hidden_dim -> the hidden state size 
            3. self.n_layers -> the number of recurrent layers. Set the default value to 1

        """
        super(LSTM, self).__init__()
        

        self.embed_len = 50
        self.hidden_dim = 75
        self.n_layers = 1
        self.p = 0.5 # default value of dropout rate
        
        self.embedding_layer = nn.Embedding(len(vocab), self.embed_len) 
        self.lstm = nn.LSTM(input_size=self.embed_len,
        hidden_size=self.hidden_dim,
        num_layers=self.n_layers,
        batch_first=True,
        bidirectional=True,
        dropout=self.p if self.n_layers > 1 else 0) 
        self.linear = nn.Linear(self.hidden_dim*2, num_classes) 
        self.dropout = nn.Dropout(self.p) 

    def forward(self, inputs, inputs_len):
        '''
        Implement the forward function to feed the input through the model and get the output.

        1. Pass the input sequences through the embedding layer to obtain the embeddings. This step should be implemented in forward_embed().
        2. Pass the embeddings through the lstm layer to obtain the output. This step should be implemented in forward_lstm().
        3. Concatenate the hidden states of the lstm as shown in the architecture diagram in HW4.ipynb. This step should be implemented in forward_concat().
        4. Pass the output from step 3 through the linear layer.

        Args:
            inputs : A (B, L) tensor containing the input sequences, where B = batch size and L = sequence length
            inputs_len :  A (B, ) tensor containing the lengths of the input sequences in the current batch prior to padding.
        Returns:
            output: Logits of each label. A tensor of size (B, C) where B = batch size and C = num_classes

        '''
        
        embedded = self.forward_embed(inputs)
        lstm_out = self.forward_lstm(embedded, inputs_len)
        final_hidden = self.forward_concat(lstm_out, inputs_len)
        final_hidden = self.dropout(final_hidden)
        output = self.linear(final_hidden)
        return output

    def forward_embed(self, inputs):
        """
        Pass the input sequences through the embedding layer.
        Args: 
            inputs : A (B, L) tensor containing the input sequences
        Returns: 
            embeddings : A (B, L, E) tensor containing the embeddings corresponding to the input sequences, where E = embedding length.
        """
        
        embeddings = self.embedding_layer(inputs)
        return embeddings
    
    def forward_lstm(self, embeddings, inputs_len):
        """
        Pack the input sequence embeddings, and then pass it through the LSTM layer to get the output from the LSTM layer, which should be padded.
        Args: 
            embeddings : A (B, L, E) tensor containing the embeddings corresponding to the input sequences.
            inputs_len : A (B, ) tensor containing the lengths of the input sequences prior to padding.
        Returns: 
            output : A (B, L', 2 * H) tensor containing the output of the LSTM. L' = the max sequence length in the batch (prior to padding) = max(inputs_len), and H = the hidden embedding size.
        
        """
        
        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, inputs_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output,_ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output
    
    def forward_concat(self, lstm_output, inputs_len):
        """
        Concatenate the forward hidden state of the last word in the sequence with the backward hidden state of the first word in the sequence.
        Take a look at the architecture diagram of our model in HW4.ipynb to visually see how this is done. Also, keep in mind the important note
        below the architecture diagram.
        Args: 
            lstm_output : A (B, L', 2 * H) tensor containing the output of the LSTM.
            inputs_len : A (B, ) tensor containing the lengths of the input sequences prior to padding.
        Returns: 
            concat : A (B, 2 * H) tensor containing the two hidden states concatenated together.
        
        """

        batch_size = lstm_output.size(0)
        hidden_dim = lstm_output.size(2)//2

        forward_indices = (inputs_len - 1).unsqueeze(1).unsqueeze(2).expand(batch_size,1,hidden_dim).to(lstm_output.device)
        forward_hidden = lstm_output.gather(1, forward_indices).squeeze(1)[:,:hidden_dim]

        backward_hidden = lstm_output[:, 0, hidden_dim:]

        concat = torch.cat((forward_hidden, backward_hidden), dim=1)
        return concat
