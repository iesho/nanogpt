import math
import torch
from torch import nn, Tensor


class IndexTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range):
        super().__init__()

        # TODO [part 2c]
        # define the encoder

        ############# YOUR CODE HERE #############
        self.encoder = nn.Embedding(n_tokens, int(d_model - 1))
        ##########################################

        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.d_model = d_model

    def forward(self, src: Tensor):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.encoder(src) * math.sqrt(self.d_model)


class IndexPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()

        # TODO [part 2c]
        # define ``positional_encoding`` as described in the GoogleDoc instruction and register it 
        #self.positional_encoding = torch.zeros(max_seq_len, 1)

        ############# YOUR CODE HERE #############
        pos = torch.arange(0, max_seq_len) / max_seq_len
        pos = pos.unsqueeze(1)
        
        ##########################################
        
        self.register_buffer('positional_encoding', pos)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """

        # TODO [part 2c]
        # concatenate ``positional_encoding`` to x (be careful of the shape)

        ############# YOUR CODE HERE #############
        c = self.positional_encoding[:x.size(0)].unsqueeze(1)
        c = c.repeat(1, x.size(1), 1)
        
        x = torch.cat((c, x), 2)
        
        ##########################################

        return self.dropout(x)
