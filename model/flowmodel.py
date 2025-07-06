import torch
import torch.nn as nn
from collections import OrderedDict
from params import par
from .MaskFlownet import *

import yaml

class Reader:
    def __init__(self, obj, full_attr=""):
        self._object = obj
        self._full_attr = full_attr
    
    def __getattr__(self, name):
        if self._object is None:
            ret = None
        else:
            ret = self._object.get(name, None)
        return Reader(ret, self._full_attr + '.' + name)

    def get(self, default=None):
        if self._object is None:
            print('Default FLAGS.{} to {}'.format(self._full_attr, default))
            return default
        else:
            return self._object

    @property
    def value(self):
        return self._object

class FlowModel(nn.Module):
    def __init__(self):
        super(FlowModel,self).__init__()

        with open(par.pretrained_flownet_path + "/kitti.yaml") as f:
            config = Reader(yaml.safe_load(f))
            
        self.flowestimate = FlowEstimate(config)

    def forward(self, input):

        feature = self.flowestimate(input)

        return feature