import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer_1 = nn.Linear(args.input_size, args.hidden_layer_size)
        self.layer_2 = nn.Linear(args.hidden_layer_size, args.output_size)
        self.dropout = MCDropout(args.dropout_rate, args.hidden_layer_size)

        if args.activation_function == "relu":
            self.act = F.relu
        else:
            self.act = F.tanh

    def forward(self, input):
        out = self.layer_1(input)
        out = self.act(out)
        out = self.dropout(out)
        out = self.layer_2(out)
        return out

    def update_dropout_masks(self):
        self.dropout.update_mask()

    def train(self, mode=True):     
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.dropout.train(mode)
        return self

    def eval(self):
        return self.train(False)


class MCDropout(nn.Module):
    def __init__(self, p_drop, layer_size):
        super().__init__()
        self.p_drop = p_drop
        self.layer_size = layer_size
        self.dropout = nn.Dropout(p = p_drop)
        self.mask = torch.diag(torch.empty((self.layer_size,)).bernoulli(1 - self.p_drop))
    
    def forward(self, input):
        #In training mode use the official dropout layer 
        if self.training:
            return self.dropout(input)
        #In evaluation mode use a static mask that only changes after each epoch
        else:
            return input @ self.mask

    def update_mask(self):
        self.mask = torch.diag(torch.empty((self.layer_size,)).bernoulli(1 - self.p_drop))
