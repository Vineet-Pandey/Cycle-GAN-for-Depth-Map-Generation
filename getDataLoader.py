
# coding: utf-8

# In[3]:


import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


# In[7]:


class getDataLoader:
    
    def __init__(self,img_type, img_dir = 'sunRGBD',
                 img_size =128,batch_size =16,num_workers=0):
        super(getDataLoader, self).__init__()
        self.img_type = img_type
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        
    
    def load_data(self):
        transform = transforms.Compose([transforms.Resize(self.img_size),transforms.RandomCrop(self.img_size),transforms.ToTensor()])
        image_path = './'+ self.img_dir
        train_path = os.path.join(image_path, self.img_type)
        test_path = os.path.join(image_path, 'test_{}'.format(self.img_type))
        train_dataset = datasets.ImageFolder(train_path, transform)
        test_dataset = datasets.ImageFolder(test_path, transform)
        
        train_loader = DataLoader(dataset = train_dataset, batch_size=self.batch_size,
                                  shuffle = True, num_workers = self.num_workers)  
        test_loader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, 
                                shuffle = True, num_workers = self.num_workers)
        return train_loader, test_loader

