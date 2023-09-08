import torch
from torch import nn
from torch.autograd import Variable
from models.utils import SelfAttention, Embedder, Decoder, Trans

class LogEncoder(nn.Module):
    def __init__(self, device, log_dropout=0, log_layer_num=4, transformer_hidden=1024, **kwargs):
        super(LogEncoder, self).__init__()
        self.hidden = kwargs["hidden_size"]
        self.self_attention = kwargs["self_attention"]
        self.window = kwargs["window_size"]
        embedding_dim = kwargs["word_embedding_dim"]

        self.net = Trans(input_size=embedding_dim, layer_num=log_layer_num, out_dim=self.hidden,
                dim_feedforward=transformer_hidden, dropout=log_dropout, device=device)
        
        if self.self_attention:
            self.attn = SelfAttention(self.hidden, kwargs["log_window_size"])
    
    def forward(self, session: torch.Tensor): #session: [batch_size, seq_num, seq_lenth, embedding_dim]
        batch_size = session.size(0)
        log_re = Variable(torch.zeros(batch_size, self.window, self.hidden, device=session.device))
        for w in range(self.window):
            seq_re = self.net(session[:, w, :, :])
            if self.self_attention: log_re[:, w, :] = self.attn(seq_re)
            else: log_re[:, w, :] = seq_re[:, -1, :]
    
        return log_re #[batch_size, window_size, hidden_size]

from torch.nn.functional import softmax as sf
class LogModel(nn.Module):
    def __init__(self, device, vocab_size=300, **kwargs):
        super(LogModel, self).__init__()
        self.feature_type = kwargs["feature_type"]
        if "word2vec" not in self.feature_type:
            self.embedder = Embedder(vocab_size, **kwargs)

        self.encoder = LogEncoder(device, **kwargs)
        self.decoder = Decoder(kwargs["hidden_size"]*kwargs["num_directions"], kwargs["window_size"], **kwargs)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_dict, flag=False):
        y = input_dict["label"].long().view(-1)
        log_x = input_dict["log_features"]
        if "word2vec" not in self.feature_type: 
            log_x = self.embedder(log_x)           
        
        log_re = self.encoder(log_x) #[batch_size, W, hidden_size*dir_num]
        logits = self.decoder(log_re)
        if y.size(0) == 1: logits = logits.unsqueeze(0)

        if flag:
            y_pred = logits.detach().cpu().numpy().argmax(axis=1)
            conf = sf(logits.detach().cpu(), dim=1).numpy().max(axis=1) #[bz, 1]
            return {"y_pred": y_pred, "conf": conf}

        loss = self.criterion(logits, y)
        return {"loss": loss}
