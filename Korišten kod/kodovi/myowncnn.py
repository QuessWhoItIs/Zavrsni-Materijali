#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
import matplotlib.pyplot as plt
import torch.optim as optim
import cv2
import matplotlib.image as imag
from matplotlib.patches import Rectangle
import time


# In[2]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3)        
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) 
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)   
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3) 
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)  
        
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)       

        self.fc1 = nn.Linear(in_features=43264, out_features=4096)        
        self.fc2 = nn.Linear(in_features=4096, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=136)
        self.fc4 = nn.Linear(in_features=136,out_features=4)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        self.bn1 = nn.BatchNorm2d(num_features=9)
        self.bn2 = nn.BatchNorm2d(num_features=18)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.bn6 = nn.BatchNorm2d(num_features=256)
        self.bn7 = nn.BatchNorm2d(num_features=256)
        self.bn8 = nn.BatchNorm2d(num_features=256)
        self.bn9 = nn.BatchNorm1d(num_features=4096)
        self.bn10 = nn.BatchNorm1d(num_features=1000)

    def forward(self, x):    
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.pool(self.bn6(F.relu(self.conv6(x))))
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.pool(self.bn8(F.relu(self.conv8(x))))

        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bn9(x)
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn10(x)
        x = self.dropout5(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout5(x)
        
        x = self.fc4(x)
        
        return x
net = Net()
net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


# In[3]:


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
        box_frame = np.array([box_frame])
        box_frame = box_frame.astype('float').reshape(-1, 2)
        size = self.frame.iloc[idx,1:3]
        sample = {'image': image,'name': self.frame.iloc[idx,0], 'size': size, 'box_frame' : box_frame}
        if self.transform:
            sample  = self.transform(sample)
        return sample
    
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
        #print(img.shape)
        return {'image': img, 'box_frame': new_box_frame, 'name' : name}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, box_frame, name = sample['image'], sample['box_frame'], sample['name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'box_frame': torch.from_numpy(np.array(box_frame)),
                'name' : name}  
class Normalize(object):
    def __call__(self,sample): #slika se normalizira pri ulitavanju sa cv2.imread()
        image, box_frame, name = sample['image'], sample['box_frame'], sample['name']
        image_copy = np.copy(image)
        box_frame_copy = np.copy(box_frame)
        #image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image_copy = image_copy/255.0
        box_frame_copy = (box_frame_copy)/300
        return {'image': image_copy, 'box_frame': box_frame_copy, 'name' : name}


# In[4]:


#df = pd.read_csv('D:/Zavrsni/MyCnn/mycsv_test.csv')
#df.dropna(axis=0, how='all',inplace=True)
#df.reset_index(drop=True)
#df.to_csv("D:/Zavrsni/MyCnn/mycsv_test2.csv", encoding='utf-8',index = False)
# resetiranje indexa unutar csv-a


# In[5]:


batch_size = 4

train_dataset = MyDataset(csv_file='D:/Zavrsni/MyCnn/mycsv_train.csv', 
                        root_dir='D:/Zavrsni/MyCnn/faces_train',
                        transform=transforms.Compose([
                            Rescale((300,300)),
                            Normalize(),
                            ToTensor()
                        ]))

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)

validation_dataset = MyDataset(csv_file='D:/Zavrsni/MyCnn/mycsv_validate.csv', 
                        root_dir='D:/Zavrsni/MyCnn/faces_validate',
                        transform=transforms.Compose([
                            Rescale((300,300)),
                            Normalize(),
                            ToTensor()
                        ]))
validation_loader = DataLoader(validation_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)

test_dataset = MyDataset(csv_file='D:/Zavrsni/MyCnn/mycsv_test.csv',
                        root_dir='D:/Zavrsni/MyCnn/faces_test',
                        transform=transforms.Compose([
                            Rescale((300,300)),
                            Normalize(),
                            ToTensor()
                        ]))


# In[6]:


lr = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)


# In[7]:


def train_net(n_epochs):
    valid_loss_min = np.inf  #početna vrijednost je beskonačno ali pošto sam morao prekidati epohe postavljam ga na posljednji validaiton loss  
    history = {'train_loss': [], 'valid_loss': [], 'epoch': []}

    for epoch in range(n_epochs):  
        train_loss = 0.0
        valid_loss = 0.0  
        net.train()
        running_loss = 0.0
        for batch_i, data in enumerate(train_loader):
            images = data['image']
            box_frame = data['box_frame']
            box_frame = box_frame.view(box_frame.size(0), -1)
            box_frame = box_frame.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)
            output_pts = net(images)
            #print(output_pts)
            #print(box_frame)
            loss = criterion(output_pts, box_frame)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.data.size(0)     
        net.eval() 
        
        with torch.no_grad():
            for batch_i, data in enumerate(validation_loader):
                images = data['image']
                box_frame = data['box_frame']
                box_frame = box_frame.view(box_frame.size(0), -1)
                box_frame = box_frame.type(torch.FloatTensor).to(device)
                images = images.type(torch.FloatTensor).to(device)
                output_pts = net(images)
                loss = criterion(output_pts, box_frame)          
                valid_loss += loss.item()*images.data.size(0) 
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(validation_loader.dataset) 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1,train_loss))
        print('Epoch: {} \tValidation loss: {:.6f}'.format(epoch+1,valid_loss))
        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))    
            torch.save(net,f'D:\\zavrsni\\epoch{epoch + 1}_loss{valid_loss}.pth')
            valid_loss_min = valid_loss
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
    print('Finished Training')
    return history


# In[8]:


net=torch.load("D:/zavrsni/epoch16_loss0.015463594878865599.pth") #učitavanje spremljenog modela

start = time.time()
n_epochs = 1
history=train_net(n_epochs)
end = time.time()
print(end-start)


# In[9]:


#prikaz greške u obliku grafa
train_loss=history['train_loss']
val_loss=history['valid_loss']
print(train_loss)
print(val_loss)
val_loss
history.keys()

epochs= range(1,len(train_loss)+1)

plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[10]:


tocni = 0
totalni_pronadeni = 0
totalni_rezultati = 0
brojac = 0
boxes = []
i=0
start = time.time()
for i in range(0,len(test_dataset),1):
    if((i+1 < len(test_dataset)) and test_dataset[i]['name'] == test_dataset[i+1]['name']):
        boxes += [test_dataset[i]['box_frame'].numpy()]
        #print("adding to box")
        continue
    brojac += 1
    boxes += [test_dataset[i]['box_frame'].numpy()]
    pixels = test_dataset[i]['image']
    pixels = pixels.type(torch.FloatTensor).to(device)
    pixels = torch.reshape(pixels,[1,3,300,300])
    faces = net(pixels)
    totalni_pronadeni += len(faces)
    pxmin,pymin,pxmax,pymax = faces[0]
    pxmin = pxmin.item()*300 #broj 300 pretstavlja unaprijed zadanu veličinu slike 
    pymin = pymin.item()*300
    pxmax = pxmax.item()*300
    pymax = pymax.item()*300
    max_iou = [0]*len(boxes)


    if(faces is None):
        boxes = []
        continue
    for j in range(len(boxes)):
        found = False
        #for k in range(len(faces[0])):
        xmin = boxes[j][0][0].item()*300
        ymin = boxes[j][0][1].item()*300
        xmax = boxes[j][1][0].item()*300
        ymax = boxes[j][1][1].item()*300
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
            print(max_iou[m])
            tocni +=1
            xmin = boxes[m][0][0].item()*300
            ymin = boxes[m][0][1].item()*300
            xmax = boxes[m][1][0].item()*300
            ymax = boxes[m][1][1].item()*300
            if(i % 100 == 0): #prikaz svakog 100-tog točnog ako postoji
                plt.imshow(test_dataset[i]['image'].transpose(0,1).transpose(1,2))
                ax = plt.gca()
                rect = Rectangle((pxmin, pymin), pxmax-pxmin, pymax-pymin, fill=False, color='blue')
                rect2 = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='red')
                ax.add_patch(rect)
                ax.add_patch(rect2)
                plt.show()
            place = i - ( len(max_iou)-1-m)
            
    boxes = []
    max_iou = []
    if (i >= 6771):
        print("prošao sam kroz lica")
        print(tocni,totalni_pronadeni,brojac)
        end = time.time()
        print(end-start)
        break

