import torch


class TransformerEncoder(torch.nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.multihead_attention = torch.nn.MultiheadAttention(hparams['embed_dim'],
                                                               hparams['num_heads'],
                                                               hparams['dropout'],
                                                               batch_first=True)
        
        self.feedforward = PositionwiseFeedForward(hparams['embed_dim'], 
                                                   hparams['feedforward_dim'],
                                                   hparams['dropout'])

        self.dropout= torch.nn.Dropout(hparams['dropout'])
        self.layernorm_attention = torch.nn.LayerNorm(hparams['embed_dim'])
        self.layernorm_feedforward = torch.nn.LayerNorm(hparams['embed_dim'])
    
    def forward(self, x):
        # Multi-Head Attention.
        attention_out, _ = self.multihead_attention(x, x, x)
        # Add & Norm
        x = self.layernorm_attention(x + self.dropout(attention_out))
        # Position wise FeedForward
        feedforward_out = self.feedforward(x)
        # Add & Norm
        x = self.layernorm_feedforward(x + self.dropout(feedforward_out))
        return x


class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."
    def __init__(self, embed_dim, feedforward_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_in = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_out = torch.nn.Linear(feedforward_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_out(self.dropout(torch.relu(self.linear_in(x))))