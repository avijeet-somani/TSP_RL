from math import sqrt

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from modules import Glimpse, GraphEmbedding, Pointer, Attention


class TSPEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=8):
        super(TSPEncoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
       

    def forward(self, x):
        x = self.transformer_layer(x)
        return x




class AttentionTSP(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_head=1,
                 C=10,
                 start_index=0):
        super(AttentionTSP, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        #self.seq_len = seq_len
        self.n_head = n_head
        self.C = C
        self.start_index = start_index
        self.embedding = GraphEmbedding(2, embedding_size)
        #self.mha = AttentionModule(embedding_size, n_head)
        self.mha =  TSPEncoder(embedding_size, n_head)     

        self.init_w = nn.Parameter(torch.Tensor(2 * self.embedding_size))
        self.init_w.data.uniform_(-1, 1)
        #self.glimpse = Glimpse(self.embedding_size, self.hidden_size, self.n_head)
        #self.pointer = Pointer(self.embedding_size, self.hidden_size, 1, self.C)
        self.pointer = Attention(self.hidden_size)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x seq_len x 2]
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs)
        h = self.mha(embedded) #Node Embedding
        h_mean = h.mean(dim=1) #Graph Embedding
        h_bar = self.h_context_embed(h_mean)
        h_rest = self.v_weight_embed(self.init_w)
        query = h_bar + h_rest

        #print('query : ' , embedded.shape, h.shape, query.shape , h_mean.shape, h_bar.shape, h_rest.shape)
        #init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
       
        for index in range(seq_len):
            
           
             #query attends to the encoder_outputs
            _, logits = self.pointer(query, h)
            _mask = mask.clone()
            #logits are scores/unnormalized log probablities
            logits[_mask] = -100000.0
            probs = torch.softmax(logits, dim=-1) #create probs along the last tensor dimension
            cat = Categorical(probs)
            chosen = cat.sample()
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            mask[[i for i in range(batch_size)], chosen] = True
            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)
