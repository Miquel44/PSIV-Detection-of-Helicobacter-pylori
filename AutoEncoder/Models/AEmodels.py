    # -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:18:03 2022

@author: debora
"""
import math
import numpy as np
import itertools

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch import Tensor
from numpy.matlib import repmat

from .models_init import *
from .NetBlocks import _CNNBlock,_UnCNNLayer



#  BACKBONE MODULES 
class Encoder(nn.Module):
    r"""Encoder class
    `".
    Input Parameters:
        1. inputmodule_params: dictionary with keys ['num_input_channels']
            inputmodule_params['num_input_channels']=Channels of input images
        2. net_params: dictionary defining architecture: 
            net_params['block_configs']: list of number of neurons for each 
            convolutional block. A block can have more than one layer
            net_params['stride']:list of strides for each block layers
            net_params['drop_rate']: value of the Dropout (equal for all blocks)
        Examples: 
            1. Encoder with 4 blocks with one layer each
            net_params['block_configs']=[[32],[64],[128],[256]]
            net_params['stride']=[[2],[2],[2],[2]]
            2. Encoder with 2 blocks with two layers each
            net_params['block_configs']=[[32,32],[64,64]]
            net_params['stride']=[[1,2],[1,2]]
            
    """

    def __init__(self, inputmodule_params,net_params):
        super().__init__()
        
        
        num_input_channels=inputmodule_params['num_input_channels']
        

            
        if 'drop_rate' in net_params.keys():  
            drop_rate=net_params['drop_rate']
        else:
            drop_rate=0
        block_configs=net_params['block_configs'].copy()
        n_blocks=len(block_configs)
        if 'stride' in net_params.keys():
            stride=net_params['stride']
        else:
            stride=[]
            for i in np.arange(len(block_configs)):
                stride.append(list(np.ones(len(block_configs[i])-1,dtype=int))+[2])
                
        # Encoder
        self.encoder=nn.Sequential(          
            )
        outchannels_encoder=[]
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i],
                stride= stride[i]               
                
            )
            self.encoder.add_module("cnnblock%d" % (i + 1), block)
            
            if stride==1:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

            num_input_channels=block_configs[i][-1] 
           # outchannels_encoder.append(num_input_channels)
           
          
               
    def forward(self, x: Tensor) -> Tensor:
        
        x=self.encoder(x)

        return x

class Decoder(nn.Module):
    r"""Decoder class
    `".
    Input Parameters:
        1. inputmodule_params: dictionary with keys ['num_input_channels']
            inputmodule_params['num_input_channels']=Channels of input images
        2. net_params: dictionary defining architecture: 
            net_params['block_configs']: list of number of neurons for each conv block
            net_params['stride']:list of strides for each block layers
            net_params['drop_rate']: value of the Dropout (equal for all blocks)
    """
    def __init__(self, inputmodule_params,net_params):
        super().__init__()
        
   
        num_input_channels=inputmodule_params['num_input_channels']
        
        self.upPoolMode='bilinear'

        if 'drop_rate' in net_params.keys():  
            drop_rate=net_params['drop_rate']
        else:
            drop_rate=0
        block_configs=net_params['block_configs'].copy()
        self.n_blocks=len(block_configs)
        
        if 'stride' in net_params.keys():
            stride=net_params['stride']
        else:
            stride=[]
            for i in np.arange(len(block_configs)):
                stride.append(list(np.ones(len(block_configs[i])-1,dtype=int))+[2])
                

        # Decoder
        self.decoder=nn.Sequential(          
            )
        
        for i0 in np.arange(self.n_blocks)[::-1]:
            i=self.n_blocks-(i0+1)
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i], 
                stride=stride[i],
                decoder=True
            )
            
            # if stride==1:
            #     self.decoder.add_module("uppool%d" % (i + 1), 
            #                               nn.Upsample(scale_factor=2, 
            #                                           mode=self.upPoolMode, align_corners=True))
            
            self.decoder.add_module("cnnblock%d" % (i0+1), block)
      

            num_input_channels=block_configs[i][-1]
        
        
        self.decoder[-1][list(self.decoder[-1].keys())[-1]].cnn_layer[2]=nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        
        input_sze=x.shape

     #   for i in np.arange(n_blocks)[::-1]:
            
        x=self.decoder(x)
    
        return x

##### GENERATIVE MODELS 

##### Variational Autoencoder
class VAECNN(nn.Module):
    r"""VAECNN model class
    `".
    Input Parameters:
        1. inputmodule_paramsEnc: dictionary with keys ['num_input_channels','dim_input']
            inputmodule_paramsEnc['num_input_channels']=Channels of input 
        2. net_paramsEnc: dictionary defining architecture of the Encoder (see Encoder class) 
        3. inputmodule_paramsDec: dictionary with keys ['num_input_channels','dim_input']
           inputmodule_paramsDec['num_input_channels']=Channels of input data
        4. net_paramsDec: dictionary defining architecture of the Encoder (see Decoder/Encoder classes) 
        5. net_paramsRep:dictionary defining architecture of the Reparametrization bottleneck:
           net_paramsRep['h_dim']=dimensionality of the flattened hidden space (=block_configs[-1][-1]*Sze of 
                                                                                Encoder(Input))
           net_paramsRep['z_dim']= dimensionality of the representation parametric space (mu,logvar for Gaussian distributed)
    """

    def __init__(self,  inputmodule_paramsEnc,net_paramsEnc,inputmodule_paramsDec,net_paramsDec,net_paramsRep):
        super().__init__()
        
        
        self.inputmodule_paramsEnc=inputmodule_paramsEnc
        self.inputmodule_paramsDec=inputmodule_paramsDec
        self.net_paramsEnc=net_paramsEnc
        self.net_paramsDec=net_paramsDec
        self.net_paramsRep=net_paramsRep
        # Encoder
        self.encoder=Encoder(inputmodule_paramsEnc,net_paramsEnc)
        # Reparametrization Layers
        self.encoder_mu = nn.Linear(net_paramsRep['h_dim'], net_paramsRep['z_dim'])
        self.encoder_logvar = nn.Linear(net_paramsRep['h_dim'], net_paramsRep['z_dim'])
        self.zRep = nn.Linear(net_paramsRep['z_dim'], net_paramsRep['h_dim'])
        # Decoder
        self.decoder=Decoder(inputmodule_paramsDec,net_paramsDec)

    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(mu.device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z

       
    def representation(self, x):
        # Sanity check for h_dim dimension
        if (self.encoder_mu.in_features!=x.shape[-1]):
            self.encoder_mu.in_features=x.shape[-1]
            self.encoder_mu.encoder_logvar=x.shape[-1]
            self.zRep.out_features=x.shape[-1]
        mu, logvar = self.encoder_mu(x), self.encoder_logvar(x)
        z = self.zRep(self.reparameterize(mu, logvar))
        
        return z, mu, logvar

    def forward(self, x: Tensor) -> Tensor:
        
        input_sze=x.shape

        x=self.encoder(x) # (Nbatch,NChannel,Sze)--> (Nbatch,LastHidden,Sze')
        x_dim=x.shape[-1]
        x=x.view(x.size(0), -1) # (Nbatch,LastHidden,Sze') --> (NBatch,LastHidden*Sze')
        z, mu, logvar =self.representation(x) # (NBatch,LastHidden*Sze')-->(NBatch,z_dim)-->(NBatch,LastHidden*Sze')
        z=z.view(z.shape[0], int(z.shape[1]/x_dim),x_dim) # (NBatch,LastHidden*Sze') -->  (Nbatch,LastHidden,Sze')
        x=self.decoder(z)
             
        return x, mu, logvar
    
# Standard Autoencoder
class AutoEncoderCNN(nn.Module):
    r"""AutoEncoderCNN model class
    `".
    Input Parameters:
        1. inputmodule_paramsEnc: dictionary with keys ['num_input_channels']
            inputmodule_paramsEnc['num_input_channels']=Channels of input images
        2. net_paramsEnc: dictionary defining architecture of the Encoder (see Encoder class) 
        3. inputmodule_paramsDec: dictionary with keys ['num_input_channels']
           inputmodule_paramsDec['num_input_channels']=Channels of input images
        4. net_paramsDec: dictionary defining architecture of the Encoder (see Decoder/Encoder classes) 
    """

    def __init__(self, inputmodule_paramsEnc,net_paramsEnc,inputmodule_paramsDec,net_paramsDec):
        super().__init__()
        
        
 
        self.inputmodule_paramsEnc=inputmodule_paramsEnc
        self.inputmodule_paramsDec=inputmodule_paramsDec
        self.net_paramsEnc=net_paramsEnc
        self.net_paramsDec=net_paramsDec
        # Encoder
        
        self.encoder=Encoder(inputmodule_paramsEnc,net_paramsEnc)
     
        # Decoder
        self.decoder=Decoder(inputmodule_paramsDec,net_paramsDec)
        
    def forward(self, x: Tensor) -> Tensor:
        
        input_sze=x.shape

        x=self.encoder(x)
        x=self.decoder(x)
       # x=F.upsample(x,size=input_sze[2::],mode=self.upPoolMode)
        
        return x

class UnetCNN(nn.Module):
    r"""OneShotCNNNet-BC model class
    `".
    """

    def __init__(self, inputmodule_params,net_params):
        super().__init__()
        
        
        num_input_channels=inputmodule_params['num_input_channels']
        self.dim=net_params['dim']
        self.upPoolMode='bilinear'
        if self.dim==3:
            self.upPoolMode='trilinear'
        drop_rate=net_params['drop_rate']
        block_configs=net_params['block_configs']
        n_blocks=len(block_configs)
        
        self.prct=None
        # Encoder
        self.encoder=nn.Sequential(          
            )
        outchannels_encoder=[]
        for i in np.arange(n_blocks):
            block = _CNNBlock(
                num_input_channels=num_input_channels,
                drop_rate=drop_rate,
                block_config=block_configs[i], 
                dim=self.dim
                
            )
            self.encoder.add_module("cnnblock%d" % (i + 1), block)
            if self.dim==2:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            elif self.dim==3:
                self.encoder.add_module("mxpool%d" % (i + 1), 
                                         nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
            num_input_channels=block_configs[i][-1] 
            outchannels_encoder.append(num_input_channels)
            
        # Decoder
        self.decoder=nn.Sequential(          
            )
        
        for i in np.arange(n_blocks)[::-1]:
            block = _CNNBlock(
                num_input_channels=num_input_channels+outchannels_encoder[i],
                drop_rate=drop_rate,
                block_config=block_configs[i][::-1], 
                dim=self.dim,
                decoder=True
            )
            self.decoder.add_module("uppool%d" % (i + 1), 
                                      nn.Upsample(scale_factor=2, 
                                                  mode=self.upPoolMode, align_corners=True))

            self.decoder.add_module("cnnblock%d" % (i + 1), block)
      

            num_input_channels=block_configs[i][0]
        
        block =  _UnCNNLayer(
            num_input_channels,
            n_neurons=inputmodule_params['num_input_channels'],
            drop_rate=drop_rate, dim=self.dim
            
         )
        self.decoder.add_module("cnnblock%d" % (i), block)
        
    def forward(self, x: Tensor) -> Tensor:
        
        input_sze=x.shape
        n_blocks=int(len(list(self.encoder.named_children()))/2)
        #encoder
        x_curr=[]
        for i in np.arange(n_blocks):
            x=self.encoder.get_submodule("cnnblock%d" % (i + 1))(x)
            x_curr.append(x)
            x=self.encoder.get_submodule("mxpool%d" % (i + 1))(x)
                   
       
        #decoder with skip connections
        for i in np.arange(n_blocks)[::-1]:
            x = self.decoder.get_submodule("uppool%d" % (i + 1))(x) 
            # diffY = x_curr[i].size()[-2] - x.size()[-2]
            # diffX = x_curr[i].size()[-1] - x.size()[-1]

            # x = F.pad(x, [diffX // 2, diffX - diffX // 2,
            #                 diffY // 2, diffY - diffY // 2])
            x=F.upsample(x,size=x_curr[i].size()[2::],mode=self.upPoolMode)
            x=torch.cat([x_curr[i], x], dim=1)
            x=self.decoder.get_submodule("cnnblock%d" % (i + 1))(x)
            
        x=self.decoder.get_submodule("cnnblock%d" % (0))(x)    

        return x    
    



