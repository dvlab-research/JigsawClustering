import torch
import os
import sys
import copy
import re

s = 'outputs/jigclu_pretrain/model_best.pth.tar'
t = 'detection/jigclu_det.pth'

s_dict = torch.load(s, map_location=torch.device('cpu'))['state_dict']
new_dict = {}
prefix = 'module.encoder.'
for k in s_dict:
    if not prefix in k:
        continue
    new_k = k[len(prefix):]
    if not 'layer' in k and not 'fc' in k:
            new_k = 'stem.' + new_k
    else:
            new_k = new_k.replace('layer1', 'res2')
            new_k = new_k.replace('layer2', 'res3')
            new_k = new_k.replace('layer3', 'res4')
            new_k = new_k.replace('layer4', 'res5')
    new_k = new_k.replace('bn1', 'conv1.norm')
    new_k = new_k.replace('bn2', 'conv2.norm')
    new_k = new_k.replace('bn3', 'conv3.norm')
    new_k = new_k.replace('bn4', 'conv4.norm')
    new_k = new_k.replace('downsample.0', 'shortcut')
    new_k = new_k.replace('downsample.1', 'shortcut.norm')
    new_k = 'backbone.bottom_up.' + new_k
    new_dict[new_k] = s_dict[k]
    print(new_k)


torch.save(new_dict, t, _use_new_zipfile_serialization=False)

