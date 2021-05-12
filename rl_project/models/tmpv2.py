import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os
from impala import ResidualBlock, ConvSequence, ImpalaCNN

class TMPNet2(nn.Module):

    def __init__(self, obs_space, num_outputs, target_width=7,
                 pooling=2, out_features = 256, conv_out_features=32,
                 log_dir=None, init_style=None, impala_layer_init=None,
                 init_all_input_channels=None,
                 d=torch.device('cuda'), grad_on=False):
        super(TMPNet2, self).__init__()
        h, w, c = obs_space.shape
        shape = (c, h, w)

        # Loading templates
        templates = {}
        image_list = os.listdir('images')
        image_list = tqdm([i for i in image_list if '.png' in i])
        for i, f in enumerate(image_list):
            if '.png' not in f:
                continue

            img = cv2.imread(os.path.join("images", f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            templates[f] = {}

            if 'robot' in f:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            fact = img.shape[1] / target_width
            templates[f]['img'] = cv2.resize(
                img, 
                (int(img.shape[1] / fact), int(img.shape[0] / fact)), 
                interpolation=cv2.INTER_CUBIC).astype(np.float32)

            # Pad for Same Shape Filter
            tmp_mat = np.zeros((7, 7, 3))
            h, w, c = templates[f]['img'].shape

            if h > target_width:
                psh, peh = 0, target_width
                ish = int(np.floor((h - target_width) * 0.5))
                ieh = ish + 7
            else:
                psh = int(np.ceil((target_width - h) * 0.5))
                peh = psh + h
                ish, ieh = 0, h
            tmp_mat[psh:peh, :, :] += templates[f]['img'][ish:ieh, :, :]
            templates[f]['img'] = tmp_mat

            #normalize by filter size and values
            templates[f]['img'] = templates[f]['img'] / (
                np.mean(templates[f]['img']) *\
                templates[f]['img'].shape[0] *\
                templates[f]['img'].shape[1])

            #zero-mean the filter
            templates[f]['img'] -= np.mean(templates[f]['img']) 

        for k, v in templates.items():
            v["fruit"] = "fruit" in k # fruit increases score
            v["food"] = "food" in k # food decreases score

        # Setting Instance Variables
        self.templates = templates
        self.target_width = target_width
        self.pooling = pooling
        self.out_feats = out_features
        self.out_conv_feats = conv_out_features

        # Constructing TempConv
        self.filter_names = []
        custom_weight = []
        for k, v in templates.items():
            self.filter_names.append(k)
            custom_weight.append(v['img'].transpose(2, 0, 1))
        custom_weight = np.array(custom_weight)
        custom_weight = torch.from_numpy(custom_weight)
        self.temp_conv = torch.nn.Conv2d(
            in_channels=3, 
            out_channels=20, 
            kernel_size=target_width,
            padding=int((target_width - 1) * 0.5),
            bias=None
        ).requires_grad_(False)
        self.temp_conv.weight.data = custom_weight.float()

        new_shape = list(obs_space.shape)
        new_shape[-1] = len(image_list) + 3 # Concats

        print(new_shape)

        self.impala = ImpalaCNN(
          obs_space=np.zeros(new_shape),
          num_outputs=num_outputs,
          first_channel=26,
          no_perm=True,
          no_scale=True
        )

    def template_match_torch(self, i, d=torch.device('cuda')):
        with torch.no_grad():
            dst = self.temp_conv(i)
        filters = dst
        return filters

    def normalize(self, x, torch_wise=True):
        mean_func = torch.mean if torch_wise else np.mean
        x = x / (mean_func(x) * x.shape[0] * x.shape[1])
        return x - mean_func(x)

    def forward(self, obs, d=torch.device('cuda')):
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        x = self.normalize(obs.float()) # obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        with torch.no_grad():
            filters = self.template_match_torch(x, d)
            
        # Concatenation of Feature Maps with Image
        x2 = obs / 255.0
        x2 = x2.permute(0, 3, 1, 2).float()
        filters = torch.cat([x2, filters], dim=1)
        return self.impala(filters)
    
    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))