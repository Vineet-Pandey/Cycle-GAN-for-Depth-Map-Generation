
# coding: utf-8

# In[1]:

# createArchitecture class is used to initialize the cycle-GAN architecture. It takes the number of channels(images channels) from the user. 
# Network can be made more deeper by inserting more residual blocks which is 6 by default in code below
# create_model takes the initial values and outputs 4 different networks, 2 generator networks and 2 discriminator networks.
# Most of the code below is self explanatory, but a comment has been provided where ever necessary

import torch
from model import Generator, Discriminator
from helper import weights_init_normal


# In[ ]:


class createArchitecture:
    
    def __init__(self, generator_channels = 3, discriminator_channels = 3, num_res_blocks = 6):
        self.generator_channels = generator_channels
        self.discriminator_channels = discriminator_channels
        self.num_res_blocks = num_res_blocks
        
    def create_model(self):
        generatorNetwork_A2B = Generator(self.generator_channels, num_ResnetBlocks = self.num_res_blocks)
        generatorNetwork_B2A = Generator(self.generator_channels, num_ResnetBlocks = self.num_res_blocks)
        discriminatorNetwork_A = Discriminator(self.discriminator_channels)
        discriminatorNetwork_B = Discriminator(self.discriminator_channels)
        
        if torch.cuda.is_available():
            generatorNetwork_A2B.cuda()
            generatorNetwork_B2A.cuda()
            discriminatorNetwork_A.cuda()
            discriminatorNetwork_B.cuda()
            print("Network moved to GPU")
        else:
            print("Network moved to CPU as no GPU devices were identified")
# Weights have been initialized by normal guassian values
        generatorNetwork_A2B.apply(weights_init_normal)
        generatorNetwork_B2A.apply(weights_init_normal)
        discriminatorNetwork_A.apply(weights_init_normal)
        discriminatorNetwork_B.apply(weights_init_normal)
        
        return generatorNetwork_A2B,generatorNetwork_B2A,discriminatorNetwork_A,discriminatorNetwork_B
        

