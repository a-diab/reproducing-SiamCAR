#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import math


# In[3]:


class Heads(nn.Module):
    def __init__(self, in_channels):
        super(Heads, self).__init__()
        
        localization_tower = []
        classification_tower = []
        num_classes=2              
        for i in range(4):            
            
            classification_tower.append(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1))
            classification_tower.append(nn.GroupNorm(32, in_channels))
            classification_tower.append(nn.ReLU())
            
            localization_tower.append(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1))
            localization_tower.append(nn.GroupNorm(32, in_channels))
            localization_tower.append(nn.ReLU())
            
        self.add_module('classification_tower', nn.Sequential(*classification_tower))
        self.add_module('localization_tower', nn.Sequential(*localization_tower))
        
        self.class_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1,padding=1)
        self.bound_box_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1,padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1,padding=1)
        
        # initialization
        
        for modules in [self.classification_tower, self.localization_tower,
                        self.class_logits, self.bound_box_pred,self.centerness]:
            
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)
                    
        # initialize the bias for focal loss
        prior_prob = 0.01                                  
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_logits.bias, bias_value)
        
    def forward(self, x):
        classification_tower = self.classification_tower(x)
        logits = self.class_logits(classification_tower)
        centerness = self.centerness(classification_tower)
        bound_box_reg = torch.exp(self.bound_box_pred(self.localization_tower(x)))

        return logits, bound_box_reg, centerness
        
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


# In[ ]:




