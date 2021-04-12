# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN
import numpy as np

def add_jigclu_cfg(cfg):
    cfg.MODEL.MOBILENETV2 = CN()

    cfg.MODEL.MOBILENETV2.OUT_FEATURES = ['m2']
    cfg.MODEL.MOBILENETV2.NORM = 'FrozenBN'


    if cfg.SEED < 0:
        cfg.SEED = np.random.randint(10000) 



