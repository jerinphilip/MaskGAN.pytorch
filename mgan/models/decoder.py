from torch import nn
import torch
import torch.nn.functional as F
import math

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size*2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
    
    def forward(self, hidden, encoder_outputs):
        Tmax, B, H = encoder_outputs.size()
        
        H = hidden.repeat(Tmax, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(H, encoder_outputs)
        return F.softmax(attn_energies, dim=0).unsqueeze(1)
    
    def score(self, hidden, encoder_outputs):
        intermediate = torch.cat([hidden, encoder_outputs], 2)
        energy = torch.tanh(self.attn(intermediate))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)
    

class AttnDecoder(nn.Module):
    def __init__(self, embedding, hidden_size, 
                 output_size, n_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.n_layers = n_layers
        self.dropout_p = dropout
        
        self.embedding = embedding
        embed_size = self.embedding.embedding_dim

        # TODO: 
        # Assert embedding check out

        self.dropout = nn.Dropout(dropout)
        self.attn = Attn(hidden_size=hidden_size)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size,
                           n_layers, dropout=dropout,
                           bidirectional=True)
        
        self.out = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, word_input, last_hidden, ctx, encoder_outputs):
        B = word_input.size(0)
        word_embedded = self.embedding(word_input)
        word_embedded = word_embedded.view(1, B, -1)
        T, B, H = word_embedded.size()
        word_embedded = self.dropout(word_embedded)
        
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        
        rnn_input = torch.cat((word_embedded, context), 2)
        output, (hidden, ctx) = self.lstm(rnn_input, (last_hidden, ctx))
        output = F.log_softmax(self.out(output), dim=2)

        return output, hidden, ctx, attn_weights.detach()
