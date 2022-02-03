import os
import zipfile 
import gdown
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re
import numpy as np
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


## Create a custom Dataset class
class CelebADataset(Dataset):
    def __init__(self, root_dir=os.path.join(CUR_DIR, '../../data/celeba'), transform=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        
        # Path to folder with the dataset
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        dataset_folder = f'{root_dir}/img_align_celeba/'
        self.dataset_folder = os.path.abspath(dataset_folder)
        if not os.path.isdir(dataset_folder):
            # URL for the CelebA dataset
            download_url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
            # Path to download the dataset to
            download_path = f'{root_dir}/img_align_celeba.zip'
            # Download the dataset from google drive
            gdown.download(download_url, download_path, quiet=False)

#             os.makedirs(dataset_folder)

            # Unzip the downloaded file 
            with zipfile.ZipFile(download_path, 'r') as ziphandler:
                ziphandler.extractall(root_dir)

        image_names = os.listdir(self.dataset_folder)

        self.transform = transform 
        image_names = natsorted(image_names)
        
        self.filenames = []
        self.annotations = []
        with open(f'{root_dir}/list_attr_celeba.txt') as f:
            for i, line in enumerate(f.readlines()):
                line = re.sub(' *\n', '', line)
                if i == 0:
                    self.header = re.split(' +', line)
                else:
                    values = re.split(' +', line)
                    filename = values[0]
                    self.filenames.append(filename)
                    self.annotations.append([int(v) for v in values[1:]])
                    
        self.annotations = np.array(self.annotations)    
              
    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the path to the image 
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img_attributes = [{-1:0, 1:1}[v] for v in self.annotations[idx]] # convert all attributes to zeros and ones
        target = torch.tensor(img_attributes).float()
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        return img, {'target': target, 'filename': img_name, 'idx': idx}
    
    
## Create a custom Dataset class
class CelebAReferenceDataset(CelebADataset):
    def __init__(self, root_dir=os.path.join(CUR_DIR, '../../data/celeba'), transform=None):
        super().__init__(root_dir, transform)
        self.filename1 = []
        self.filename2 = []
        self.domain = []
        self.domain_names = self.header
        for domain_idx, domain in enumerate(self.domain_names):
            domain_indices_1 = np.argwhere(self.annotations[:, domain_idx]>0).squeeze()
            domain_indices_2 = np.argwhere(self.annotations[:, domain_idx]<0).squeeze()
            for idx1, idx2 in zip(domain_indices_1, np.random.choice(domain_indices_2, len(domain_indices_1))):
                self.filename1.append(self.filenames[idx1])
                self.filename2.append(self.filenames[idx2])
                self.domain.append(domain_idx)
            for idx1, idx2 in zip(domain_indices_2, np.random.choice(domain_indices_1, len(domain_indices_2))):
                self.filename1.append(self.filenames[idx1])
                self.filename2.append(self.filenames[idx2])
                self.domain.append(domain_idx)
            
            
    def __len__(self):
        return len(self.domain)
    
    def __getitem__(self, idx):
        # Get the path to the image 
        
        domain_idx = self.domain[idx]
        domain_name = self.domain_names[domain_idx]
        img_name1, img_name2 = self.filename1[idx], self.filename2[idx]
        img_path1, img_path2 = [os.path.join(self.dataset_folder, img_name) for img_name in [img_name1, img_name2]]
        
        # Load image and convert it to RGB
        img1, img2 = [Image.open(img_path).convert('RGB') for img_path in [img_path1, img_path2]]
        # Apply transformations to the image
        if self.transform:
            img1, img2 = [self.transform(img) for img in [img1, img2]]
        return img1, img2, {'domain': domain_idx, 'domain_name': domain_name, 'filename1': img_name1, 'filename2': img_name2, 'idx': idx}
