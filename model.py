'''
Created on May 6, 2018
@author: jana
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepVS(nn.Module):
    '''
    Implements DeepVS neural network architecture
    '''
    def __init__(self, vocab_size, embedding_dim, cf, h, kc, kp):
        super(DeepVS, self).__init__()
        self.embeddings_context = nn.Embedding(vocab_size, embedding_dim) 
        self.linear1 = nn.Linear(((kc+kp) * 3 + kp ) * embedding_dim, cf)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(cf,h)
        self.relu2 = nn.ReLU(True)
        self.linear3 = nn.Linear(h, 2)
        self.dropout = nn.Dropout(.5, False)
        
        
        
    def forward(self, inputs, mask):
        mbSize = inputs.size()[0]
        embeddings  = self.embeddings_context(inputs.view(-1, inputs.size()[2]))
        out = self.linear1(embeddings.view(embeddings.size()[0], -1))
        out = self.relu1(out)
        out = out.view(mbSize, out.size()[0]/mbSize, out.size()[1]) # puts minibatch dimension back
        out = out + mask.view(mask.size()[0], mask.size()[1], 1)
        out = torch.max(out, dim = 1)[0]
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.linear3(out) 
        log_probs = F.log_softmax(out)
        return log_probs