import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, NLLLoss
from torch.nn import functional as F
import io

def GeLU(x):
    return x * 0.5 * (1.0 + torch.erf(x/math.sqrt(2.0)))

def bert_layer_norm(x, weight, bias):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + 1e-12)
    return weight * x + bias

def embedding(x, x_mask, x_token, params, train, dropout_prob):
    batch_size, sent_len = x.size()
    x_pos = torch.arange(sent_len).unsqueeze(0).expand([batch_size, -1]).cuda()

    x = F.embedding(x, params[0]) * x_mask.unsqueeze(-1).float()
    x_pos = F.embedding(x_pos, params[1])
    x_token = F.embedding(x_token, params[2])

    x = x + x_pos + x_token 

    x = bert_layer_norm(x, params[3], params[4])
    x = F.dropout(x, p = dropout_prob, training=train)

    return x

def encoding_layer(x, attention_mask, params, train, dropout_prob):
    q_w, q_b, k_w, k_b, v_w, v_b, d_w, d_b, sln_w, sln_b, f1_w, f1_b, f2_w, f2_b, fln_w, fln_b = params

    attention_head_size = int(768 / 12)

    #attention
    q = F.linear(x, q_w, q_b)
    k = F.linear(x, k_w, k_b)
    v = F.linear(x, v_w, v_b)

    shape = q.size()[:-1] + (12, attention_head_size)
    q, k, v = [var.view(*shape).permute(0, 2, 1, 3) for var in (q, k, v)]

    attention_scores = torch.matmul(q, k.transpose(3, 2))
    attention_scores = attention_scores / math.sqrt(attention_head_size)
    attention_scores = attention_scores + attention_mask

    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    attention_probs = F.dropout(attention_probs, p = dropout_prob, training=train)

    y = torch.matmul(attention_probs, v)
    y = y.permute(0, 2, 1, 3).contiguous()

    shape = x.size()
    y = y.view(*shape)

    y = F.linear(y, d_w, d_b)
    y = F.dropout(y, p = dropout_prob, training=train)
    x = bert_layer_norm(x + y, sln_w, sln_b)

    #ff layer
    y = F.linear(x, f1_w, f1_b)
    y = GeLU(y)
    y = F.linear(y, f2_w, f2_b)
    y = F.dropout(y, p = dropout_prob, training=train)
    x = bert_layer_norm(x + y, fln_w, fln_b)
    return x

def encoder(x, attention_mask, params, train, dropout_prob):
    for i in range(12):
        x = encoding_layer(x, attention_mask, params[i*16:(i+1)*16], train, dropout_prob)
    return x

def first_pooler(x, w, b, train, dropout_prob):
    x = x[:, 0] 
    x = F.linear(x, w, b)
    x = F.tanh(x)
    return x

class MTDNNModel(nn.Module):
    def __init__(self, train_embedding, hidden_size, vocab_size, max_len, type_vocab_size, num_layers, num_attention_heads, filter_size, num_labels, dropout_prob=0.1, task_clusters=None):
        super(MTDNNModel, self).__init__()
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob
        self.task_clusters = task_clusters
        self.train_embedding = train_embedding
        
    def forward(self, params, task_id, x, x_mask, x_token, y=None, train=True):

        attention_mask = x_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_mask = attention_mask.float()

        if self.train_embedding:
            x = embedding(x, x_mask, x_token, params[:5], train, self.dropout_prob)
        else:
            with torch.no_grad():
                x = embedding(x, x_mask, x_token, params[:5], train, self.dropout_prob)

        x = encoder(x, attention_mask, params[5:16*12+5], train, self.dropout_prob)
        x = first_pooler(x, params[5+16*12], params[5+16*12+1], train, self.dropout_prob)

        x = F.dropout(x, p = self.dropout_prob, training=train)
        x = F.linear(x, params[5+16*12+2+task_id*2], params[5+16*12+2+task_id*2+1])

        if not train:
            return x

        x = x.view(-1, self.num_labels[task_id]) if self.num_labels[task_id] > 1 else x.view(-1)
        return F.cross_entropy(x, y.view(-1)) if self.num_labels[task_id] > 1 else F.mse_loss(x, y.view(-1))
