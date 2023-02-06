import torch
from torch import nn
from torch.autograd import Variable
import math

class Decoder(nn.Module):
    def __init__(self, encoded_dim, T, **kwargs):
        super(Decoder, self).__init__()
        linear_size = kwargs["linear_size"]

        layers = []
        for i in range(kwargs["decoder_layer_num"]-1):
            input_size = encoded_dim if i == 0 else linear_size
            layers += [nn.Linear(input_size, linear_size), nn.ReLU()]
        layers += [nn.Linear(linear_size, 2)]
        self.net = nn.Sequential(*layers)
        
        self.self_attention = kwargs["self_attention"]
        if self.self_attention:
            self.attn = SelfAttention(encoded_dim, T)

    def forward(self, x: torch.Tensor): #[batch_size, T, hidden_size*dir_num]
        if self.self_attention: ret = self.attn(x)
        else: ret = x[:, -1, :]
        return self.net(ret)

class Embedder(nn.Module):
    def __init__(self, vocab_size=300, **kwargs):
        super(Embedder, self).__init__()
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.embedder = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
    
    def forward(self, x): #[batch_size, T, word_lst_num]
        return self.embedder(x.long())

class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):
        # x: [batch_size, window_size, 2*hidden_size]
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

class Trans(nn.Module):
    def __init__(self, input_size, layer_num, out_dim, dim_feedforward=512, dropout=0, device="cpu", norm=None, nhead=8):
        super(Trans, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=input_size, 
                            dim_feedforward=dim_feedforward, #default: 2048
                            nhead=nhead, 
                            dropout=dropout, 
                            batch_first=True)
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=layer_num, norm=norm).to(device)
        self.out_layer = nn.Linear(input_size, out_dim)
    def forward(self, x: torch.Tensor): #[batch_size, T, var]
        out = self.net(x)
        return self.out_layer(out)