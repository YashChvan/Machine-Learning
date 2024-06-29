import torch
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from transformers import BertModel

class Encoder(nn.Module):
    def __init__(self,enc_hid_dim:int,freeze = True):
        super(Encoder,self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        if(freeze):
            for param in self.bert_model.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        outputs = self.bert_model(**x)
#         print(outputs.last_hidden_state.shape)
        hidden = torch.zeros(1, x["input_ids"].shape[0], 150).to("cuda")
#         print(hidden.shape)
        cell = torch.zeros(1, x["input_ids"].shape[0], 150).to("cuda")
        output = outputs.last_hidden_state.permute(1,0,2)
        return output,hidden,cell

class Attention(nn.Module):
    def __init__(self,enc_hid_out_dim: int,dec_hid_dim: int):
        super().__init__()
        self.enc_hid_dim = enc_hid_out_dim
        self.dec_hid_dim = dec_hid_dim
        self.energy = nn.Linear(self.dec_hid_dim+self.enc_hid_dim,1)
        self.sfmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self,dec_hidden ,enc_states):
        seq_len = enc_states.shape[0]
        enc_state = enc_states #.repeat((dec_hidden.shape[0],1,1))
#         print(dec_hidden.shape,enc_state.shape)
        dec_hidden_reshaped = dec_hidden.repeat((seq_len,1,1))
#         print(dec_hidden_reshaped.shape,enc_state.shape)
        energy = self.relu(self.energy(torch.cat((dec_hidden_reshaped,enc_state),dim=2)))
        attension = self.sfmax(energy)
        attension = attension.permute(1,2,0)
        enc_st = enc_state.permute(1,0,2)
        context_vec = torch.bmm(attension,enc_st).permute(1,0,2)
#         print(context_vec.shape)
        return context_vec


class Decoder(nn.Module):
    def __init__(self,embedding , input_dim:int, emb_dim:int, hid_dim:int, output_dim:int, num_layers:int, dropout: float, attension):
        super(Decoder,self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.p_dropout = dropout
        self.dropout = nn.Dropout(self.p_dropout)
        self.embedding = embedding
        self.rnn = nn.LSTM(768+ self.emb_dim,self.hidden_dim,self.num_layers)
        self.fc = nn.Linear(self.hidden_dim,self.output_dim)
        self.attn = attension
        
    def forward(self,x,enc_states,hidden,cell):
#         print("decoder")
#         print("x:",x.shape)
        x = x.unsqueeze(0)
#         print("x seqz:",x.shape)
        emb = self.dropout(self.embedding(x))
#         print("emb:",emb.shape)

        context_vec = self.attn(hidden,enc_states)
#         print(context_vec.shape,emb.shape)
        rnn_inpt = torch.cat((context_vec,emb),dim=2)
        
        outputs, (hidden,cell) = self.rnn(rnn_inpt,(hidden,cell))
#         print(" out, hidden, cell: ",outputs.shape,hidden.shape,cell.shape)
        pred = self.fc(outputs)
#         print("fc:" ,pred.shape)
        pred = pred.squeeze(0)
#         print("fc seq:" ,pred.shape)
        return pred, hidden,cell

class Seq2Seq_Attn_bert(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device, teacher_forcing_ratio: float = 0.6):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,src: Tensor,trg: Tensor=None, evalu = False) -> Tensor:

        batch_size = src["input_ids"].shape[0]
        mx_ln = trg.shape[0] if trg is not None else 100
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(mx_ln, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> token
        x = trg[0,:] if trg is not None else torch.tensor([2]).to(self.device)
        teacher_force = random.random() < self.teacher_forcing_ratio
        for t in range(1, mx_ln):
            inp = x
            output, hidden, cell = self.decoder(x,encoder_outputs, hidden, cell)
            outputs[t] = output
            b_guess = output.argmax(1)
            teacher_force = random.random() < self.teacher_forcing_ratio
            x = (trg[t] if (teacher_force and not evalu) else b_guess)
        return outputs