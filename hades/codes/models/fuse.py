import torch
import torch.nn as nn
from models.utils import Embedder, Decoder
from models.kpi_model import KpiEncoder
from models.log_model import LogEncoder

class CrossAttention(nn.Module): #k=V
    def __init__(self, dimensions):
        super(CrossAttention, self).__init__()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context): #[batch_size, length, dim]
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        query = query.reshape(batch_size * output_len, dimensions)
        query = self.linear_in(query)
        query = query.reshape(batch_size, output_len, dimensions)
       
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())  # (batch_size, output_len, query_len)

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context) # (batch_size, output_len, dimensions)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class MultiEncoder(nn.Module):
    def __init__(self, var_nums, device, vocab_size=300, fuse_type="cross_attn", **kwargs):
        super(MultiEncoder, self).__init__()
        self.hidden_size = kwargs["hidden_size"]
    
        self.feature_type = kwargs["feature_type"]
        if "word2vec" not in self.feature_type:
            self.embedder = Embedder(vocab_size, **kwargs)
        self.log_encoder = LogEncoder(device, **kwargs)
        
        self.kpi_encoder = KpiEncoder(var_nums, device, **kwargs) 
        
        self.fuse_type = fuse_type
        if self.fuse_type == "cross_attn" or self.fuse_type == "sep_attn":
            self.attn_alpha = CrossAttention(self.hidden_size)
            self.attn_beta = CrossAttention(self.hidden_size)
    
    def forward(self, kpi_x, log_x):
        if "word2vec" not in self.feature_type:
            log_x = self.embedder(log_x)    

        kpi_re = self.kpi_encoder(kpi_x) #[batch_size, T, hidden_size]
        log_re = self.log_encoder(log_x) #[batch_size, W, hidden_size]
        fused = None
        if self.fuse_type == "cross_attn":
            fused_alpha, _ = self.attn_alpha(query=log_re, context=kpi_re)
            fused_beta, _ = self.attn_beta(query=kpi_re, context=log_re)
            fused = torch.cat((fused_alpha, fused_beta), dim=1)
        elif self.fuse_type == "sep_attn":
            fused_kpi, _ = self.attn_alpha(query=kpi_re, context=kpi_re)
            fused_log, _ = self.attn_beta(query=log_re, context=log_re)
            fused = torch.cat((fused_kpi, fused_log), dim=1)
        elif self.fuse_type == "concat":
            fused = torch.cat((kpi_re, log_re), dim=1)
        
        return fused, (kpi_re, log_re) #[batch_size, T+W, hidden_size]

class JoinDecoder(nn.Module):
    def __init__(self, encoded_dim, T, **kwargs):
        super(JoinDecoder, self).__init__()
        self.logd = Decoder(encoded_dim, T, **kwargs)
        self.kpid = Decoder(encoded_dim, T, **kwargs)
    
    def forward(self, kpi_re, log_re):
        kpi_logits = self.kpid(kpi_re)
        log_logits = self.logd(log_re)
        assert kpi_logits.shape == log_logits.shape
        return kpi_logits, log_logits

from torch.nn.functional import softmax as sf
class MultiModel(nn.Module):
    def __init__(self, var_nums, device, fuse_type="cross_attn", **kwargs):
        super(MultiModel, self).__init__()
        self.fuse_type = fuse_type
        self.encoder = MultiEncoder(var_nums=var_nums, device=device, fuse_type=fuse_type, **kwargs)

        self.hidden_size = kwargs["hidden_size"]
        self.window = kwargs["window_size"]
        if self.fuse_type == "join":
            self.decoder = JoinDecoder(self.hidden_size, self.window, **kwargs)
        else:
            self.decoder = Decoder(self.hidden_size, self.window*2, **kwargs)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_dict, flag=False):
        y = input_dict["label"].long().view(-1)
        bz = y.size(0)
        
        fused_re, (kpi_re, log_re) = self.encoder(input_dict["kpi_features"], input_dict["log_features"]) 
        
        if self.fuse_type == "join": #[batch_size, out_dim]
            kpi_logits, log_logits = self.decoder(kpi_re, log_re)
            logits = (kpi_logits + log_logits) /2
            if bz == 1:
                kpi_logits = kpi_logits.unsqueeze(0)
                log_logits = log_logits.unsqueeze(0)
                logits = logits.unsqueeze(0)
            if flag: 
                kpi_pred = kpi_logits.detach().cpu().numpy().argmax(axis=1)
                log_pred = log_logits.detach().cpu().numpy().argmax(axis=1)
                y_pred = kpi_pred | log_pred
                conf = sf(logits.detach().cpu(), dim=1).numpy().max(axis=1) #[bz, 1]
                return {"y_pred": y_pred, "conf": conf}
            loss = self.criterion(logits, y)
            return {"loss":loss} # Training
        else:
            logits = self.decoder(fused_re)
            if bz == 1: logits = logits.unsqueeze(0)
            if flag:
                y_pred = logits.detach().cpu().numpy().argmax(axis=1)
                conf = sf(logits.detach().cpu(), dim=1).numpy().max(axis=1) #[bz, 1]
                return {"y_pred": y_pred, "conf": conf}
            loss = self.criterion(logits, y)
            return {"loss":loss} # Training
            
            
            
        
        
        