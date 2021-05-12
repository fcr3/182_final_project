import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import cv2
import os
from .impala import ResidualBlock, ConvSequence, ImpalaCNN

class TMPNet_template_init(nn.Module):

    def __init__(self, obs_space, num_outputs, target_width=7,
                 pooling=2, out_features = 256, conv_out_features=32,
                 proc_conv_ksize=3, proc_conv_stride=2,
                 gpu=None, impala_layer_init=-1, 
                 init_style='resize', log_dir='./log_ADAM',
                 init_all_input_channels=False, grad_on=None):
        super(TMPNet_template_init, self).__init__()

        assert(init_style in ['resize', 'fragment'])

        if gpu is not None and gpu >= 0 and torch.cuda.is_available():
            # assert torch.cuda.is_available()
            print('Using gpu...')
            d = torch.device("cuda:{}".format(gpu))
        else:
            if torch.cuda.is_available():
                print('Considering using GPU b/c it is availabe!')
            print('Using cpu...')
            d = torch.device("cpu")

        h, w, c = obs_space.shape
        shape = (c, h, w)

        # Loading templates
        templates = {}
        image_list = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images'))
        image_list = tqdm([i for i in image_list if '.png' in i])
        for i, f in enumerate(image_list):
            if '.png' not in f:
                continue
            
            f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', f)

            templates[f] = {}
            img = cv2.imread(os.path.join("images", f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if 'robot' in f:
                img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            fact = img.shape[1] / target_width
            img= cv2.resize(
                img, 
                (int(img.shape[1] / fact), int(img.shape[0] / fact)), 
                interpolation=cv2.INTER_CUBIC).astype(np.float32)
              
            # Pad for Same Shape Filter
            tmp_mat = np.zeros((7, 7, 3))
            h, w, c = img.shape

            if h > target_width:
                psh, peh = 0, target_width
                ish = int(np.floor((h - target_width) * 0.5))
                ieh = ish + 7
            else:
                psh = int(np.ceil((target_width - h) * 0.5))
                peh = psh + h
                ish, ieh = 0, h
            tmp_mat[psh:peh, :, :] += img[ish:ieh, :, :]
            img = tmp_mat

            
            #resize everything to a 3x3 for the impala kernel
            if init_style == 'resize':
              templates[f]['img'] = cv2.resize(img, (3, 3), 
              interpolation=cv2.INTER_CUBIC).astype(np.float32)
            #fragment the templates
            else:
              for i in range(3):
                for k in range(3):
                  img_name = 'fragment_' + str(i) + '_' + str(k)
                  templates[f][img_name] = img[2*i:2*i+3,2*k:2*k+3,:]
                  
        for k, v in templates.items():
            v["fruit"] = "fruit" in k # fruit increases score
            v["food"] = "food" in k # food decreases score

        # Setting Instance Variables
        self.templates = templates
        self.target_width = target_width
        self.pooling = pooling
        self.out_feats = out_features
        self.out_conv_feats = conv_out_features

        self.init_all_input_channels = init_all_input_channels

        #can't init all input channels since its already done with layer 0
        assert(not (init_all_input_channels and impala_layer_init == 0))

        #cant fragment since theres only 3 input channels per output with layer 0
        assert(not (init_style == 'fragment' and impala_layer_init == 0))

        #we need to init over all input channels with fragments, since that makes it easier to implement
        assert(not (init_style == 'fragment' and not init_all_input_channels))

        # Constructing template tensor
        if init_style == 'resize':
          self.filter_names = []
          custom_weight = []
          for k, v in templates.items():
              self.filter_names.append(k)
              custom_weight.append(v['img'].transpose(2, 0, 1))
          custom_weight = np.array(custom_weight)
          self.templates_tensor = torch.from_numpy(custom_weight)
          self.templates_tensor = self.templates_tensor.to(d).float()
        else:
          self.filter_names = []
          custom_weight = []
          for k, v in templates.items():
              self.filter_names.append(k)
              for fragment_name, fragment in v.items():
                if 'fragment' in fragment_name:
                  custom_weight.append(fragment.transpose(2, 0, 1))
          custom_weight = np.array(custom_weight)
          self.templates_tensor = torch.from_numpy(custom_weight)
          self.templates_tensor = self.templates_tensor.to(d).float()

        with torch.no_grad():
          for i in range(self.templates_tensor.shape[0]):
            for k in range(self.templates_tensor.shape[1]):
              u, s, v = torch.svd(self.templates_tensor[i, k, :, :])
              self.templates_tensor[i,k,:,:] = u

        print(self.templates_tensor.shape)

        self.impala = ImpalaCNN(
          obs_space=obs_space,
          num_outputs=num_outputs,
          first_channel=26,
          no_perm=True,
          no_scale=True
        )

        self.templates = templates

        self.impala_layer_init = impala_layer_init
        self.init_style = init_style

        self.log_dir = log_dir

        print('initing impala layer', impala_layer_init, 'with style', init_style)

        self.template_init(self.impala, impala_layer_init, init_style, self.templates_tensor)

    def template_init(self, impala, impala_layer_init, init_style, templates_tensor):
      with torch.no_grad():
        for name, param in impala.named_parameters():
          if 'conv' in name and 'weight' in name:
            if impala_layer_init == 0:
              out_c, in_c, _, _ = param.shape
              if init_style == 'resize':
                self.out_idx = np.random.choice(out_c, templates_tensor.shape[0], replace=False)
              else:
                self.out_idx = np.random.choice(out_c, int(templates_tensor.shape[0] / 9), replace=False)
              
              if self.init_all_input_channels:
                if init_style == 'resize':
                  self.in_idx = np.random.choice(3, in_c)
                else:
                  self.in_idx = np.random.choice(27, in_c) #9 filters, 3 channels each
              else:
                self.in_idx = np.random.choice(in_c, 3, replace=False)

              print('in_idx', self.in_idx)

              self.templated_name = name

              print('init impala layer')

              #if the layer is 0, we need to set in_idx to 2,1,0
              #since templates are BGR and input is RGB
              if (self.impala_layer_init == 0):
                self.in_idx = np.array([0, 1, 2])

              with open(os.path.join(self.log_dir, 'select_indices.txt'), 'w') as f:
                f.write(str(self.out_idx))
                f.write(str(self.in_idx))

              if init_style == 'resize':
                for idx in range(self.templates_tensor.shape[0]):
                  if self.init_all_input_channels:
                    for idx_c in range(self.in_idx.shape[0]):
                        param[self.out_idx[idx], idx_c].copy_(templates_tensor[idx,self.in_idx[idx_c],:,:])
                  else:
                    for idx_c in range(3):
                      param[self.out_idx[idx], self.in_idx[idx_c], :, :].copy_(templates_tensor[idx,idx_c,:,:])
              else:
                for idx in range(int(self.templates_tensor.shape[0] / 9)):
                  for idx_c in range(self.in_idx.shape[0]):
                    param[self.out_idx[idx], idx_c].copy_(templates_tensor[idx * 9 + int(self.in_idx[idx_c] / 3), self.in_idx[idx_c] % 3, :,:])
              return
            else:
              impala_layer_init -= 1
      
      print('no layers set in impala')


    def forward(self, obs):
        assert obs.ndim == 4,  f'Invalid Shape: {obs.shape}'
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        return self.impala(x)
    
    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def visualize_templates(self):

      if self.init_style == 'resize':
        templates_numpy = self.templates_tensor.numpy()

        fig, ax = plt.subplots(12, 2, figsize=(20, 20))
        i = 0
        idx_templates = 0
        for k, v in self.templates.items():
          if 'food' in k or 'fruit' in k:
            ax[i, 0].imshow(templates_numpy[idx_templates].transpose(1, 2, 0))
            ax[i, 0].set_title(k + " kernel")
            ax[i, 1].imshow(v['img'] / 255)
            ax[i, 1].set_title(k + " img")

            i += 1
          idx_templates += 1
        
        plt.show()
      
      else:
        templates_numpy = self.templates_tensor.numpy()

        fig, ax = plt.subplots(12, 9, figsize=(40, 40))
        i = 0
        idx_templates = 0
        for k, v in self.templates.items():
          if 'food' in k or 'fruit' in k:
            idx_sub = 0
            for frag_name, frag in v.items():
              if 'fragment' in frag_name:
                ax[i, idx_sub].imshow(frag / 255)
                ax[i, idx_sub].set_title(k + " img frag " + frag_name)
                idx_sub += 1
            i += 1
          idx_templates += 1
        
        fig.tight_layout()
        plt.show()
    
    def kernel_norm(self):
      norms = {}

      if (self.impala_layer_init == -1):
        for i in range(15):
          norms[i] = 0
      else:
        norms[self.impala_layer_init] = 0

      with torch.no_grad():
        if (self.impala_layer_init == -1):
          idx = 0
          curr = self.impala.state_dict()
          for name, param in curr.items():
            if 'conv' in name and 'weight' in name:
              for i in range(self.templates_tensor.shape[0]):
                for k in range(self.templates_tensor.shape[1]):
                  norms[idx] += torch.norm(param[i,k]).item()
              norms[idx] /= (self.templates_tensor.shape[0] * self.templates_tensor.shape[1])
              idx += 1
        else:
          if self.init_all_input_channels:
            for i in range(self.out_idx.shape[0]):
              for k in range(self.in_idx.shape[0]):
                curr = self.impala.state_dict()[self.templated_name][self.out_idx[i],k]
                norm = torch.norm(curr)
                norms[self.impala_layer_init] += norm.item()
            norms[self.impala_layer_init] /= (self.out_idx.shape[0] * self.in_idx.shape[0])
          else:
            for i in range(self.out_idx.shape[0]):
              for k in range(self.in_idx.shape[0]):
                curr = self.impala.state_dict()[self.templated_name][self.out_idx[i],self.in_idx[k]]
                norm = torch.norm(curr)
                norms[self.impala_layer_init] += norm.item()
            norms[self.impala_layer_init] /= (self.out_idx.shape[0] * self.in_idx.shape[0])

      return norms



    def kernel_diff_norm(self):
      avgs = {}

      if (self.impala_layer_init == -1):
        for i in range(15):
          avgs[i] = 0
      else:
        avgs[self.impala_layer_init] = 0

      with torch.no_grad():
        if (self.impala_layer_init == -1):
          idx = 0
          curr = self.impala.state_dict()
          for name, param in curr.items():
            if 'conv' in name and 'weight' in name:
              for i in range(self.templates_tensor.shape[0]):
                for k in range(self.templates_tensor.shape[1]):
                  avgs[idx] += torch.norm(param[i,k] - self.templates_tensor[i,k]).item()
              avgs[idx] /= (self.templates_tensor.shape[0] * self.templates_tensor.shape[1])
              idx += 1
        else:
          if self.init_style == 'resize':
            if self.init_all_input_channels:
              for i in range(self.out_idx.shape[0]):
                for k in range(self.in_idx.shape[0]):
                  curr = self.impala.state_dict()[self.templated_name][self.out_idx[i],k]
                  norm = torch.norm(curr - self.templates_tensor[i,self.in_idx[k]])
                  avgs[self.impala_layer_init] += norm.item()
              avgs[self.impala_layer_init] /= (self.out_idx.shape[0] * self.in_idx.shape[0])
            else:
              for i in range(self.out_idx.shape[0]):
                for k in range(self.in_idx.shape[0]):
                  curr = self.impala.state_dict()[self.templated_name][self.out_idx[i],self.in_idx[k]]
                  norm = torch.norm(curr - self.templates_tensor[i,k])
                  avgs[self.impala_layer_init] += norm.item()
              avgs[self.impala_layer_init] /= (self.out_idx.shape[0] * self.in_idx.shape[0])
              
          else:
            for i in range(self.out_idx.shape[0]):
              for k in range(self.in_idx.shape[0]):
                curr = self.impala.state_dict()[self.templated_name][self.out_idx[i], k]
                norm = torch.norm(curr - self.templates_tensor[i * 9 + int(self.in_idx[k] / 3), self.in_idx[k] % 3, :, :])
                avgs[self.impala_layer_init] += norm.item()
            avgs[self.impala_layer_init] /= (self.out_idx.shape[0] * self.in_idx.shape[0])

      return avgs