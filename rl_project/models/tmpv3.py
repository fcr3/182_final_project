import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os
from .impala import ResidualBlock, ConvSequence, ImpalaCNN

class TMPNet3(nn.Module):

    def __init__(self, obs_space, num_outputs, target_width=7,
                 pooling=2, out_features = 256, conv_out_features=32,
                 proc_conv_ksize=3, proc_conv_stride=2,
                 log_dir=None, init_style=None, impala_layer_init=None,
                 init_all_input_channels=None,
                 gpu=None, grad_on=False):
        super(TMPNet3, self).__init__()
        h, w, c = obs_space.shape
        shape = (c, h, w)

        if gpu is not None and gpu >= 0 and torch.cuda.is_available():
            # assert torch.cuda.is_available()
            print('Using gpu...')
            d = torch.device("cuda:{}".format(gpu))
        else:
            if torch.cuda.is_available():
                print('Considering using GPU b/c it is availabe!')
            print('Using cpu...')
            d = torch.device("cpu")

        # Loading templates
        templates = {}
        image_list = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images'))
        image_list = tqdm([i for i in image_list if '.png' in i])
        for i, f in enumerate(image_list):
            if '.png' not in f:
                continue

            f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', f)

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
        self.templates_tensor = torch.from_numpy(custom_weight)
        self.templates_tensor = self.templates_tensor.to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ).float()
        
        self.proc_conv = torch.nn.Conv2d(
            in_channels=3, 
            out_channels=3, 
            kernel_size=proc_conv_ksize,
            stride=proc_conv_stride,
            bias=None
        )

        new_shape = list(obs_space.shape)
        new_shape[-1] = len(templates)
        self.impala = ImpalaCNN(
          obs_space=np.zeros(new_shape),
          num_outputs=num_outputs,
          first_channel=26,
          no_perm=True,
          no_scale=True
        )

    def forward(self, obs):
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        filters = self.proc_conv(self.templates_tensor) # 20 x 3 x 3 x 3
        filters = torch.tanh(filters) # Transform to [-1, 1]
        x = F.conv2d(x, filters, padding=1) # b x 20 x 64 x 64
        return self.impala(x)
    
    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))