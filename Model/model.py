import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from Resnet50_Backbone import resnet50
from heads import Heads

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # backbone
        self.backbone = resnet50([2, 3, 4])

        # head
        self.car_head = Heads(256)