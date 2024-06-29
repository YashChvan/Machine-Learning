import random
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,embedding,input_dim:int, emb_dim:int, enc_hid_dim:int, num_layers:int, dropout:float):
        super(Encoder,self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = enc_hid_dim
        self.prob_dropout = dropout
        self.num_layers = num_layers

        self.dropout = nn.Dropout(self.prob_dropout)
        self.embedding = embedding
        self.rnn = nn.LSTM(self.emb_dim,self.hidden_dim,self.num_layers,dropout = self.prob_dropout,bidirectional=True)


    def forward(self, x: Tensor):
        emb = self.dropout(self.embedding(x))
        outputs, (hidden,cell) = self.rnn(emb)
        return outputs,hidden,cell



class Decoder(nn.Module):
    def __init__(self,embedding , input_dim:int, emb_dim:int, hid_dim:int, output_dim:int, num_layers:int, dropout: float):
        super(Decoder,self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.p_dropout = dropout

        self.dropout = nn.Dropout(self.p_dropout)
        self.embedding = embedding
        self.rnn = nn.LSTM(self.emb_dim,self.hidden_dim,self.num_layers,dropout = self.p_dropout)
        self.fc = nn.Linear(self.hidden_dim,self.output_dim)


    def forward(self,x,enc_opt,hidden,cell):
        x = x.unsqueeze(0)
        emb = self.dropout(self.embedding(x))
        outputs, (hidden,cell) = self.rnn(emb,(hidden,cell))
        pred = self.fc(outputs)
        pred = pred.squeeze(0)
        return pred, hidden,cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device, teacher_forcing_ratio: float = 0.6):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,src: Tensor,trg: Tensor=None, evalu = False) -> Tensor:

        batch_size = src.shape[1]
        mx_ln = trg.shape[0] if trg is not None else 100
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(mx_ln, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        x = trg[0,:]
        
        for t in range(1, mx_ln):
            output, hidden, cell = self.decoder(x,encoder_outputs, hidden, cell)
            outputs[t] = output
            b_guess = output.argmax(1)
            teacher_force = random.random() < self.teacher_forcing_ratio
            x = (trg[t] if (teacher_force and not evalu) else b_guess)
        return outputs
    
