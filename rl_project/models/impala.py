import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os

class ResidualBlock(nn.Module):

    def __init__(self, channels, k_size=3):
        # Padding originally 1

        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=k_size,
                               padding=int((k_size - 1) * 0.5))
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=k_size,
                               padding=int((k_size - 1) * 0.5))

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels, k_size=3):
        # Padding originally 1

        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=k_size,
                              padding=int((k_size - 1) * 0.5))
        self.max_pool2d = nn.MaxPool2d(kernel_size=k_size,
                                       stride=2,
                                       padding=int((k_size - 1) * 0.5))
        self.res_block0 = ResidualBlock(self._out_channels, k_size)
        self.res_block1 = ResidualBlock(self._out_channels, k_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2
        
class ImpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs, first_channel=16, k_size=3,
                 distill_net=None, corrs=[], no_perm=False, no_scale=False):

        # corrs array holds the indices of the conv sequences
        # that we will cache for passing to the distill_net

        super(ImpalaCNN, self).__init__()
        self.no_perm = no_perm
        self.no_scale = no_scale
        self.corrs = corrs
        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [first_channel, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels, k_size=k_size)
            
            # Building Student Network
            if distill_net:
                distill_net.append(
                  SmallConvSequence(shape, out_channels, k_size=k_size),
                  conv_seq.get_output_shape()
                )

            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)

        # Compiling Student Network
        if distill_net:
            distill_net.compile()

        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def assign_distill_net(self, distill_net):
        # Useful for assignment AFTER LOADING
        self.distill_net = distill_net

    def forward_distill(self, obs, cl_func, kl_func):
        # print('Using large model forward_distill')
        # cl_func: correspondence loss function (probably MSE)
        # kl_func: KL-Diverence loss function
        assert self.distill_net != None, "Distill net not set..."
        distill_net = self.distill_net

        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        if not self.no_scale:
            x = obs / 255.0  # scale to 0-1
            x_dist = obs / 255.0
        else:
            x = obs
            x_dist = obs * 1

        if not self.no_perm:
            x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
            x_dist = x_dist.permute(0, 3, 1, 2)

        corr_loss = 0
        for i, conv_seq in enumerate(self.conv_seqs):
            with torch.no_grad():
                x = conv_seq(x)
            
            x_dist = distill_net(x_dist, i=i)
            if len(self.corrs) != 0 and i in self.corrs:
                corr_loss += cl_func(x_dist, x)
        
        with torch.no_grad():
          x = torch.flatten(x, start_dim=1)
          x = torch.relu(x)
          x = self.hidden_fc(x)
          x = torch.relu(x)
          logits = self.logits_fc(x)
          dist = torch.distributions.Categorical(logits=logits)
          value = self.value_fc(x)

        x_dist = torch.flatten(x_dist, start_dim=1)
        x_dist = torch.relu(x_dist)
        x_dist = distill_net.hidden_fc(x_dist)
        x_dist = torch.relu(x_dist)
        logits_dist = distill_net.logits_fc(x_dist)
        dist_dist = torch.distributions.Categorical(logits=logits_dist)
        value_dist = distill_net.value_fc(x_dist)

        kl_loss = kl_func(logits_dist, logits)
        return None, None, dist_dist, value_dist, corr_loss, kl_loss

    def forward(self, obs):
        # print('Using large model forward')
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        if not self.no_scale:
            x = obs / 255.0  # scale to 0-1
        else:
            x = obs

        if not self.no_perm:
            x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        for i, conv_seq in enumerate(self.conv_seqs):
            x = conv_seq(x)
        
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

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))