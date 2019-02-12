
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class ResnetBlock(nn.Module):
    def __init__(self,input_features):
        super(ResnetBlock, self).__init__()
        Resnet_convolution = [
                              nn.ReflectionPad2d(1),
                              nn.Conv2d(input_features,input_features,3),
                              nn.InstanceNorm2d(input_features),
                              nn.LeakyReLU(negative_slope = 0.01),
                              nn.ReflectionPad2d(1),
                              nn.Conv2d(input_features, input_features,3),
                              nn.InstanceNorm2d(input_features)
                             ]
        self.Resnet_convolution = nn.Sequential(*Resnet_convolution)
    
    def forward(self, x):
        return x+self.Resnet_convolution(x)


# In[3]:


class Generator(nn.Module):
    def __init__(self, input_channels, num_ResnetBlocks = 6):
        
        super(Generator,self).__init__()
        
        GeneratorArchitecture = [
                                 nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_channels,64,7),
                                 nn.InstanceNorm2d(64),
                                 nn.LeakyReLU(negative_slope = 0.01, inplace = True)
                                ]
        
        input_features = 64
        output_features = input_features*2
        
        for _ in range(2):
            GeneratorArchitecture += [ 
                                       nn.Conv2d(input_features, output_features,3,stride =2, padding =1),
                                       nn.InstanceNorm2d(output_features),
                                       nn.PReLU(init =0.25)
                                     ]
            
            input_features = output_features
            output_features = input_features*2
        
        for _ in range(num_ResnetBlocks):
            GeneratorArchitecture += [ResnetBlock(input_features)]
        
        output_features = input_features//2
        
        for _ in range(2):
            GeneratorArchitecture += [ 
                                      nn.ConvTranspose2d(input_features, output_features,3, stride =2, padding=1, output_padding =1),
                                      nn.InstanceNorm2d(output_features),
                                      nn.PReLU(init =0.25) 
                                     ]
            input_features= output_features
            output_features = input_features//2
            
        
        GeneratorArchitecture += [nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_features, input_channels,7),
                                 nn.Tanh()]
        
        self.GeneratorArchitecture = nn.Sequential(*GeneratorArchitecture)
    def forward(self, x):
        return self.GeneratorArchitecture(x)
        


# In[4]:


class Discriminator(nn.Module):
    
    def __init__(self, input_channels):
        
        super(Discriminator,self).__init__()
        
        DiscriminatorArchitecture = [ 
                                 nn.Conv2d(input_channels, 128, 4, stride =2, padding=1),
                                 nn.LeakyReLU(negative_slope = 0.01, inplace = True),
                                 ]
        DiscriminatorArchitecture += [ 
                                  nn.Conv2d(128, 256,4,stride=2, padding =1),
                                  nn.InstanceNorm2d (256),
                                  nn.LeakyReLU(negative_slope = 0.01, inplace = True)
                                  ]
        DiscriminatorArchitecture += [ 
                                  nn.Conv2d(256,512,3, padding =1),
                                  nn.InstanceNorm2d (256),
                                  nn.LeakyReLU(negative_slope = 0.01, inplace = True)
                                  ]
        DiscriminatorArchitecture += [nn.Conv2d(512, 1, 4, padding =1)]
    
        self.DiscriminatorArchitecture = nn.Sequential(*DiscriminatorArchitecture)
        
    
    def forward(self, x):
        
        x = self.DiscriminatorArchitecture(x)
        return x
        
    
    
   
    
    
    
    
    

                                
    
    
    
    
    

