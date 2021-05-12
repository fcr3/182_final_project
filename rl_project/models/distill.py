import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os

class SmallConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels, k_size=3):
        # Padding originally 1
        super(SmallConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=k_size,
                              padding=int((k_size - 1) * 0.5))
        self.max_pool2d = nn.MaxPool2d(kernel_size=k_size,
                                       stride=2,
                                       padding=int((k_size - 1) * 0.5))

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        return torch.relu(x)

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class SmallCNN(nn.Module):
    def __init__(self, num_outputs, mid_feats=256, T=0.5, alpha=0.5,
                 cl_func=lambda x, y: 0):
        super(SmallCNN, self).__init__()
        self.conv_seqs = nn.ModuleList()
        self.num_outputs = num_outputs
        self.last_shape = None
        self.mid_feats = mid_feats
        self.T = T
        self.alpha = alpha
        self.cl_func = cl_func
    
    def append(self, module, output_shape):
        self.conv_seqs.append(module)
        self.last_shape = output_shape

    def compile(self):
        assert self.last_shape != None, "Haven't appended anything to modlist"

        num_outputs = self.num_outputs
        shape = self.last_shape
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=self.mid_feats)
        self.logits_fc = nn.Linear(in_features=self.mid_feats, 
                                   out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=self.mid_feats, 
                                  out_features=1)
    
    def forward(self, x, i=-1, normalize=True, permute=True):
        # print('Using small model forward')
        if i != -1:
            assert i < len(self.conv_seqs), f"Oob for passed in index: {i}"
            return self.conv_seqs[i](x)
        
        if normalize:
            x = x / 255.0
        if permute:
            x = x.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.conv_seqs):
            x = conv(x)
        
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)

        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path, d=None):
        if d == None:
            self.load_state_dict(torch.load(model_path))