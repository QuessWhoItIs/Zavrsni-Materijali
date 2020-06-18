#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
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
import tkinter as tk
from PIL import ImageTk, Image
import threading
import time

detector = MTCNN(select_largest=False, post_process=False, device='cuda') #učitavanje mtcnn modela na grafičkoj
plt.ion()
df = pd.read_csv('D:/Zavrsni/annotations.csv')
IMAGES_PATH = 'D:/Zavrsni/faces'
unique_files = df.file_name.unique()
classes = df.class_name.unique().tolist()
print(df.shape)


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
        image = plt.imread(img_name)
        box_frame = self.frame.iloc[idx,3:7]
        size = self.frame.iloc[idx,1:3]
        sample = {'image': image,'name': self.frame.iloc[idx,0], 'size': size, 'box_frame' : box_frame}
        if self.transform:
            sample  = self.transform(sample)
        return sample
        


# In[3]:


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
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


# In[4]:


new_dataset = MyDataset(csv_file='D:/Zavrsni/mycsv.csv', 
                        root_dir='D:/Zavrsni/faces',
                        transform=transforms.Compose([
                            Rescale((300,300)),
                            ToTensor()
                        ]))
boxes = []
wrongboxes = []

i=0
for i in range(0,len(new_dataset),1):
    if((i+1 < len(new_dataset)) and new_dataset[i]['name'] == new_dataset[i+1]['name']):
        boxes += [new_dataset[i]['box_frame'].numpy()]
        #print("adding to box")
        continue
    boxes += [new_dataset[i]['box_frame'].numpy()]
    pixels = new_dataset[i]['image'].transpose(0,1).transpose(1,2).numpy()
    pixels = pixels * 255
    faces = detector.detect(pixels)
    max_iou = [0]*len(boxes)


    if(faces[0] is None):
        boxes = []
        correcting = {'name': new_dataset[i]['name'], 'box': new_dataset[i]['box_frame'].numpy(), 'index': i}
        wrongboxes.append(correcting) 
        continue
        
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
        if(max_iou[m] < 0.2):
            place = i - ( len(max_iou)-1-m)
            correcting = {'name': new_dataset[place]['name'], 'box': new_dataset[place]['box_frame'].numpy(), 'index': place}
            wrongboxes.append(correcting)
                
    
    boxes = []
    max_iou = []
    if(i % 1000 == 0):
        print(i)
    if (i >= 3000):
        print("prošao sam kroz lica")
        print("broj točno pronađenih lica = ", tocni)
        print("broj ukupno pronadenih lica = ", ukupno_pronadenih)
        print("broj ukupnih lica u skupu = ", stvarna_kolicina)
        
        break


# In[5]:


#print(wrongboxes)
#print(len(wrongboxes))


# In[6]:


starter = ()
ending_pos = ()
delete = False


class correctingclass():
    def __init__(self, master, name,box):
        self.imgpath = name
        self.pilimg = Image.open(self.imgpath)
        self.img = self.pilimg.resize((300,300)) #korišten magični broj 300
        self.canvas = tk.Canvas(master, width=self.img.size[0],height=self.img.size[1])  
        
        self.ciimg = ImageTk.PhotoImage(self.img)  
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.ciimg) 
        self.canvas.create_rectangle(box[0],box[1],box[2],box[3],outline="red")
        self.canvas.bind("<Button-1>", self.callback)
        self.canvas.bind("<Button-3>", self.callback2)
        self.canvas.pack()
        self.B = tk.Button(master, text ="confirm", command = self.buttonCallBack)
        self.B.pack()   
        self.C = tk.Button(master, text ="box is ok, keep it", command = self.buttonCallBackOk)
        self.C.pack()
        self.D = tk.Button(master, text ="delete that box", command = self.buttonDelete)
        self.D.pack()
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.counter = 0
        
    def buttonDelete(event):
        global delete
        delete = True
        result_available.set()
        master.destroy()
        
    def callback(self,event):
        global starter
        if self.counter==0:
            self.x1 = event.x
            self.y1 = event.y
            print(event.x,event.y)
            starter = (event.x,event.y)
            self.counter += 1
        else:
            print("use mouse button 3")

    def callback2(self,event):
        global ending_pos
        if self.counter>0:
            self.x2 = event.x
            self.y2 = event.y
            print(event.x,event.y)
            ending_pos = (event.x,event.y)
            self.canvas.create_rectangle(self.x1,self.y1,self.x2,self.y2,outline="green")
            self.counter=0
        else:
            print("use mouse button 1")
            
    
    def buttonCallBackOk(event):
        result_available.set()
        print("moving on")
        master.destroy()
        
    def buttonCallBack(event):
        global starter
        global ending_pos
        result_available.set()
        if(starter is not () and ending_pos is not ()):
            print("confirmed")
            print(starter, ending_pos)
            master.destroy()
        else:
            print("please select the box first")
        
def threadstarter(master,name,box):
    k1 = correctingclass(master,name,box)
    

for wrong in wrongboxes:
    delete = False
    result_available = threading.Event()
    image_df = df.loc[wrong['index']]
    print(image_df, "correcting this image")
    print(wrong['box'],"correcting this box")
    print(wrong['box'][0]*df.loc[wrong['index']].width/300,
          wrong['box'][2]*df.loc[wrong['index']].width/300,
          wrong['box'][1]*df.loc[wrong['index']].height/300,
          wrong['box'][3]*df.loc[wrong['index']].height/300)

        
    starter = ()
    ending_pos = ()
    name = IMAGES_PATH + '/' + wrong['name'] 
    box = wrong['box']
    print(box,"ovo lice")
    counter = 0
    master = tk.Tk()
    thread = threading.Thread(target = threadstarter(master,name,box))
    master.mainloop()
    result_available.wait()
    print("continuing", starter,ending_pos)
    
    if(delete):
        print("goodbye box")
        df = df.drop(wrong['index'],axis=0)
        continue
        
    if(starter is not () and ending_pos is not () ):
        if(starter[0] < ending_pos[0]):
            nxmin = starter[0]
            nxmax = ending_pos[0]
        if(starter[0] >= ending_pos[0]):
            nxmax = starter[0]
            nxmin = ending_pos[0]
        if(starter[1] < ending_pos[1]):
            nymin = starter[1]
            nymax = ending_pos[1]
        if(starter[1] >= ending_pos[1]):
            nymax = starter[1]
            nymin = ending_pos[1]
        #korišten magični broj 300...dolje je resize definiran na 300,300    
        cxmin = df.loc[wrong['index']].width/300 * nxmin
        cxmax = df.loc[wrong['index']].width/300 * nxmax
        cymin = df.loc[wrong['index']].height/300 * nymin
        cymax = df.loc[wrong['index']].height/300 * nymax
        correctedresult = [cxmin,cymin,cxmax,cymax]
        #treba ih skalirati na originalnu velicinu
        print(correctedresult)
        df.loc[wrong['index']:wrong['index'], "x_min":"y_min"] = cxmin,cymin
        df.loc[wrong['index']:wrong['index'], "x_max":"y_max"] = cxmax,cymax
        print("image corrected")
        image_df = df.loc[wrong['index']]
        print(image_df)
    
    else:
        print("original box is ok")
        image_df = df.loc[wrong['index']]
        print(image_df)
        


# In[7]:


#when its done you should reset index and save
df.reset_index(drop=True)
df.to_csv("D:/Zavrsni/mycsv.csv", encoding='utf-8',index = False)
