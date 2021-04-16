#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
import torch.nn as nn
import numpy as np 
import torchvision.transforms as transforms 
import torchvision 
import torch.nn.functional as F


# In[ ]:


class Basic_Block(nn.Module):
    expansion = 1
    def __init__(self,input_planes,output_planes,stride=1,dim_change=None):     #dim_change : downsample
        super(Basic_Block,self).__init__()
        #convolutional layers with batch norms
        self.conv1 = nn.Conv2d(input_planes,output_planes,stride=stride,kernel_size=3,padding=1,bias=False,)
        self.bn1   = nn.BatchNorm2d(output_planes)
        self.conv2 = nn.Conv2d(output_planes,output_planes,stride=1,kernel_size=3,padding=1)
        self.bn2   = nn.BatchNorm2d(output_planes)
        self.dim_change = dim_change
    
    def forward(self,x):
        
        res = x  #Save residual
        
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)  #downsample
        
        output += res
        output = F.relu(output)

        return output


# In[ ]:


class bottleNeck(nn.Module):
    expansion = 4    # to expand the number of channels by 4 
    def __init__(self,input_planes,output_planes,stride=1,dim_change=None):
        super(bottleNeck,self).__init__()

        self.conv1 = nn.Conv2d(input_planes,output_planes,kernel_size=1,stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_planes)
        self.conv2 = nn.Conv2d(output_planes,output_planes,kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_planes)
        self.conv3 = nn.Conv2d(output_planes,output_planes*self.expansion,kernel_size=1, bias=False) # the output *4
        self.bn3 = nn.BatchNorm2d(output_planes*self.expansion)
        self.dim_change = dim_change      
        
    def forward(self,x):
        res = x
        
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))

        if self.dim_change is not None:
            res = self.dim_change(res)
        
        output += res
        output = F.relu(output)
        return output


# In[ ]:


class ResNet(nn.Module):
    def __init__(self,block,num_layers,used_layers):
        super(ResNet,self).__init__()
        # according to research paper:
        self.input_planes = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=0, bias=False)
        self.bn1   = torch.nn.BatchNorm2d(64)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # diff
         
        self.layer1 = self._layer(block,64,num_layers[0],stride=1)
        self.layer2 = self._layer(block,128,num_layers[1],stride=2)
        self.layer3 = self._layer(block,256,num_layers[2],stride=2)
        self.layer4 = self._layer(block,512,num_layers[3],stride=2)
        self.averagePool = torch.nn.AvgPool2d(kernel_size=4,stride=1)
        self.fc    =  torch.nn.Linear(512*block.expansion,used_layers)

    def _layer(self,block,output_planes,num_layers,stride=1):
        dim_change = None
        if stride!=1 or input_planes!= self.output_planes*block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes,output_planes*block.expansion,kernel_size=1,stride=stride),
                                             nn.BatchNorm2d(output_planes*block.expansion))
        netLayers =[]
        netLayers.append(block(self.input_planes,output_planes,stride=stride,dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1,num_layers):
            netLayers.append(block(self.input_planes,output_planes))
            self.input_planes = planes * block.expansion
        
        return nn.Sequential(*netLayers)
    
    def forward(self,x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x1)                 # diff

        p1 = self.layer1(x)
        p2= self.layer2(p1)
        p3= self.layer3(p2)
        p4= self.layer4(p3)

        
        output = [x1, p1, p2, p3, p4]
        output = [output[i] for i in self.used_layers]
        if len(output) == 1:
            return output[0]
        else:
            return output # return p4


# In[ ]:


def resnet50(used_layers):
   ## Constructs a ResNet-50 model.
    model = ResNet(Bottleneck, [3, 4, 6, 3], used_layers)
    return model


# In[ ]:





# In[ ]:





# In[ ]:






    
    

    

