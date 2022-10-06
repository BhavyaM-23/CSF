import numpy as np
import pandas as pd
import cv2
import os
import gc
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm.auto import tqdm
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
from glob import glob
import warnings
from scipy import ndimage
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from PIL import Image
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from skimage import exposure
import kornia
import kornia.augmentation as augmentation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from kaggle_volclassif.utils import interpolate_volume
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import shutil

from . models import uploadfile

# 3D convolutional neural network
class Conv3DNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=0)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.norm1 = nn.BatchNorm3d(num_features=16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.norm2 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.norm3 = nn.BatchNorm3d(num_features=64)
        self.avg = nn.AdaptiveAvgPool3d((7, 1, 1))
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=448, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=8)
        
    def forward(self, x):
        # Conv block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm1(out)
        
        # Conv block 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm2(out)
        
        # Conv block 3
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.norm3(out)
         # Average & flatten
        out = self.avg(out)
        out = self.flat(out)
        
        # Fully connected layer
        out = self.lin1(out)
        out = self.relu(out)
        
        # Output layer (no sigmoid needed)
        out = self.lin2(out)
        
        return out

class RSNADataset(Dataset):
    # Initialise
    def __init__(self,):
        super().__init__()
        
        # Image paths
        self.volume_dir = './resources/vol'
        self.l=[entry for entry in os.listdir(self.volume_dir) if os.path.isfile(os.path.join(self.volume_dir, entry))]
    
    # Get item in position given by index
    def __getitem__(self, index):
        
        # load 3d volume
        patient=os.listdir(self.volume_dir)[index]
        path = os.path.join(self.volume_dir,patient)
        #print(path)
        vol = torch.load(path).to(torch.float32)
        
        return (vol.unsqueeze(0), patient[:-3])

    # Length of dataset
    def __len__(self):
        return len(self.l)

# Convert dicom images to 3d tensor
def convert_volume(dir_path, out_dir = "test_volumes", size = (224, 224, 224)):
    ls_imgs = glob(os.path.join(dir_path, "*.dcm"))
    ls_imgs = sorted(ls_imgs, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    imgs = []
    for p_img in ls_imgs:
        dicom = pydicom.dcmread(p_img)
        img = apply_voi_lut(dicom.pixel_array, dicom)
        img = cv2.resize(img, size[:2], interpolation=cv2.INTER_LINEAR)
        imgs.append(img.tolist())
    vol = torch.tensor(imgs, dtype=torch.float32)

    vol = (vol - vol.min()) / float(vol.max() - vol.min())
    vol = interpolate_volume(vol, size).numpy()
    vol = exposure.equalize_adapthist(vol, kernel_size=np.array([64, 64, 64]), clip_limit=0.01)
    # vol = exposure.equalize_hist(vol)
    vol = np.clip(vol * 255, 0, 255).astype(np.uint8)
    
    path_pt = os.path.join(out_dir, f"{os.path.basename(dir_path)}.pt")
    torch.save(torch.tensor(vol), path_pt)

def pred():
    if 'vol' in os.listdir('./resources'):
        shutil.rmtree('./resources/vol')
    os.mkdir("./resources/vol")
    convert_volume('./media','./resources/vol')
    #f=open(os.path(os.listdir('./resources/vol')[0]))
    #volume(f)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device
    model = Conv3DNet().to(device)

    # Load checkpoint
    PATH='./resources/Conv3DNet.pt'
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH)
    else:
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))

    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_loss = checkpoint['val_loss']

    # Evaluation mode
    model.eval()
    model.to(device)

    test_dataset = RSNADataset()
    # Dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        # Loop over batches
        for i, (imgs, patient) in enumerate(test_loader):
            # Send to device
            imgs = imgs.to(device)
            
            # Make predictions
            preds = model(imgs)
            
            # Apply sigmoid
            sig = nn.Sigmoid()
            preds = sig(preds)
            preds = preds.to('cpu')
            p=list(preds.numpy().squeeze())
            #print("Patient {0} ".format(patient))
        return p