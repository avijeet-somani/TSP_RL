import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from modules import Attention, GraphEmbedding



class LSTMTSP(nn.Module) : 
    def __init__(self, embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            start_index
            ):
        super(LSTMTSP, self).__init__()  
        self.n_glimpses = 1
        embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.embedding = GraphEmbedding(2, embedding_size)
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        #print('decoder input dummy data: ', self.decoder_start_input.data)
        self.encoder = nn.LSTM(embedding_size, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, self.hidden_size, batch_first=True)
        #self.glimpse = Attention(self.hidden_size)
        self.pointer = Attention(self.hidden_size)

    
    def forward(self, inputs) : 
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        #print(batch_size, seq_len)
        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded) #encoder_outputs hold the hidden state for all time-steps . Output shape : encoder_outputs :-(batch, seq-length, hidden-size), hidden :- (batch, hidden-size)
        
        prev_chosen_logprobs = []
        preb_chosen_indices = []
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        num_nodes_to_explore = seq_len
        if self.start_index == None : 
            decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        else : 
            batch_start_index = torch.full((batch_size,), self.start_index )
            decoder_input = embedded[torch.arange(batch_size), batch_start_index, :]
            mask[[i for i in range(batch_size)], batch_start_index] = True
            num_nodes_to_explore = seq_len - 1
            preb_chosen_indices.append(batch_start_index)

        #print('Shapes : encoder_outputs.shape, hidden.shape, context.shape , mask.shape , decoder_input.shape ' , encoder_outputs.shape, hidden.shape, context.shape , mask.shape , decoder_input.shape )
        for index in range(num_nodes_to_explore):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
        
            query = hidden.squeeze(0)
           

            #query attends to the encoder_outputs
            _, logits = self.pointer(query, encoder_outputs)


            _mask = mask.clone()
            #logits are scores/unnormalized log probablities
            logits[_mask] = -100000.0
            probs = torch.softmax(logits, dim=-1) #create probs along the last tensor dimension
            cat = Categorical(probs) 
            chosen = cat.sample()
            #print('categorical , chosen ', cat, chosen.shape)
            mask[[i for i in range(batch_size)], chosen] = True
            log_probs = cat.log_prob(chosen)
            decoder_input = embedded[torch.arange(batch_size), chosen, :] #chose the embeddings for the chosen index
            
            #print('decoder_input : ', decoder_input.shape)
            prev_chosen_logprobs.append(log_probs)
            preb_chosen_indices.append(chosen)
        
        return torch.stack(prev_chosen_logprobs, 1), torch.stack(preb_chosen_indices, 1)





