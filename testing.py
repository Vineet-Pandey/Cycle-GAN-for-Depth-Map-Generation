
# coding: utf-8

# In[37]:
# This python script has been developed to infer the results the cycle-GAN  model created in the model.py file. 
# This script uses the trained weights of discriminators and the generators stored in checkpoints_cyclegan folder

from model import Generator
import torch
from torchvision.utils import save_image
from createArchitecture import createArchitecture
from getDataLoader import getDataLoader


# In[38]:

# Model creation using the createArchitecture class. create_model() creates the model. 
# More description of the architecture creation has been described in the createArchitecture.py script
torch.set_default_tensor_type('torch.DoubleTensor')
mod = createArchitecture()
# mod = mod.double()
# Since this deals with generation of depth map from the RGB image, discriminators are not considered. However,
# create_model outputs 2-Generators and 2-discriminators
gen_A2B,gen_B2A,_,_ = mod.create_model() 
# gen_A2B = gen_A2B.cuda()
# gen_B2A = gen_B2A.cuda() 


# In[39]:

# Loading the pre-trained weights
gen_A2B.load_state_dict(torch.load('checkpoints_cyclegan/gen_A2B.pkl'))
gen_B2A.load_state_dict(torch.load('checkpoints_cyclegan/gen_B2A.pkl'))


# In[40]:

# the architecture is evaluated and put on the CUDA Device. However, in the absence of a device, it can 
# run on CPU as well.

gen_A2B.eval()
gen_B2A.eval()

gen_A2B = gen_A2B.cuda()
gen_B2A = gen_B2A.cuda() 


# In[41]:


f =getDataLoader('rgb', batch_size=1)
_, test_dataloader_A = f.load_data()
test_iter = iter(test_dataloader_A)


# In[42]:
#testing the images in the dataset

cuda0 = torch.device('cuda:0')
test_rgb_image= test_iter.next()[0]
test_rgb_image = torch.tensor(test_rgb_image,dtype=torch.double, device=cuda0)

save_image(test_rgb_image, 'output/depth/test.png')


# In[43]:


generated_depth_image=gen_A2B(test_rgb_image).data
generated_depth_image = (generated_depth_image)*-1.5
generated_depth_image.size()


# In[44]:


save_image(generated_depth_image, 'output/depth/depth.png')


# In[45]:


torch.cuda.current_device()

