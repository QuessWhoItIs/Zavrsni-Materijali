#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from torchvision import transforms, utils
from skimage import io,transform
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN


# In[2]:


class MyDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()            
        img_name = os.path.join(self.root_dir,self.frame.iloc[idx,0])
        image = io.imread(img_name)
        box_frame = self.frame.iloc[idx,3:7]
        size = self.frame.iloc[idx,1:3]
        sample = {'image': image,'name': self.frame.iloc[idx,0], 'size': size, 'box_frame' : box_frame}
        if self.transform:
            sample  = self.transform(sample)
        return sample
    
class Rescale(object):
    """
        Unosi se željeni tuple za promijenu veličine slike
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, size, box_frame, name = sample['image'], sample['size'], sample['box_frame'], sample['name']
        new_box_frame = []

        h, w = image.shape[:2]
        h2,w2 = size
        min = box_frame[:2] #x1,y1
        max = box_frame[2:] #x2,y2
        #print('first is h,w  second is min,max', h,w,min,max)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped because for images,
        # x and y axes are axis 1 and 0 respectively
        max = max * [new_w / w,new_h / h]
        min = min * [new_w / w,new_h / h]
        new_box_frame[0:2] = min
        new_box_frame[2:4] = max
        #print(new_box_frame)
        
        #print('first is new_h,new_w  second is new_min,new_max', new_h,new_w,min,max)

        return {'image': img, 'box_frame': new_box_frame, 'name' : name}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, box_frame, name = sample['image'], sample['box_frame'], sample['name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print(image)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'box_frame': torch.from_numpy(np.array(box_frame)),
                'name' : name}    


# In[3]:


detector = MTCNN(select_largest=False, post_process=False, device='cuda')


# In[4]:


new_dataset = MyDataset(csv_file='D:/Zavrsni/mycsv.csv', 
                        root_dir='D:/Zavrsni/faces',
                        transform=transforms.Compose([
                            Rescale((300,300)),
                            ToTensor()
                        ]))

boxes = []
tocni = 0
ukupno_pronadenih = 0
stvarna_kolicina = len(new_dataset)

for i in range(0,len(new_dataset),1):
    if((i+1 < len(new_dataset)) and new_dataset[i]['name'] == new_dataset[i+1]['name']):
        boxes += [new_dataset[i]['box_frame'].numpy()]
        #print("adding to box")
        continue
    boxes += [new_dataset[i]['box_frame'].numpy()]
    pixels = new_dataset[i]['image'].transpose(0,1).transpose(1,2).numpy()
    pixels = pixels * new_dataset[i]['image'].size()[2]
    faces = detector.detect(pixels)
    max_iou = [0]*len(boxes)


    if(faces[0] is None):
        boxes = []
        continue
        
    ukupno_pronadenih += len(faces[0])
    for j in range(len(boxes)):
        found = False
        for k in range(len(faces[0])):
            pxmin,pymin,pxmax,pymax = faces[0][k]
            xmin,ymin,xmax,ymax = boxes[j]
            interxmin = max(xmin,pxmin)
            interymin = max(ymin,pymin)
            interxmax = min(xmax,pxmax)
            interymax = min(ymax,pymax)
            inter_area = max(0, interxmax - interxmin + 1) * max(0, interymax - interymin + 1)
            actual_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            pred_area = (pxmax - pxmin + 1) * (pymax - pymin + 1)
            iou = inter_area / float(actual_area + pred_area - inter_area)
            if(iou > 0):
                if(max_iou[j] < iou):
                    max_iou[j] = iou
     
    for m in range(len(max_iou)):
        if(max_iou[m] > 0.5):
            tocni += 1
                
    
    boxes = []
    max_iou = []
    if(i % 1000 == 0):
        print(i)
    if (i >= 16884):
        print("prošao sam kroz lica")
        print("broj točno pronađenih lica = ", tocni)
        print("broj ukupno pronadenih lica = ", ukupno_pronadenih)
        print("broj ukupnih lica u skupu = ", stvarna_kolicina)
        
        break

