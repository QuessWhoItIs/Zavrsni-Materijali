#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv2
from matplotlib.patches import Rectangle
from matplotlib import pyplot
import os
import mtcnn
import shutil
import PIL.Image as Image
import pandas as pd


# In[2]:


index = 12822 #ovo je isto kao i
brojac = 0
i = 12822  # pri pokretanju ovaj broj je jednak broju zadnje slike koja je spremljena u dataset
izadi = False
data = {}
dataset = []


# In[3]:


x = 0
izadi = False
for root,subd,fil in os.walk("D:lfw"):
    for one in fil:
        if(one.endswith(".jpg")):
            index = index + 1
            brojac = brojac + 1
            filename = root + '/' + one
            print(brojac)
            img = Image.open(filename)
            img = img.convert('RGB')
            img.save("C:/Users/Mario/Desktop/faces/face_" + str(index) + ".jpeg", "JPEG")
            img_width, img_height = img.size
            pixels = pyplot.imread(filename)
            # create the detector, using default weights
            detector = mtcnn.MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(pixels)
            for face in faces:
                data = {}
                x,y,width,height = face['box']
                data['file_name'] = "face_" + str(index) + ".jpeg"
                data['width'] = img_width
                data['height'] = img_height
                data["x_min"] = x
                data["y_min"] = y
                data["x_max"] = x + width
                data["y_max"] = y + height
                data['class_name'] = 'face'
                if(brojac % 825 == 0):
                    if(brojac == 1001):
                        izadi = True
                    print(brojac)
                if(brojac >= 500):
                    dataset.append(data)
                    print("brojac preko 500, spremamo csv")
                    df = pd.DataFrame(dataset)
                    df.to_csv('C:/Users/Mario/Desktop/annotations.csv', mode='a', header=False, index = None)
                    dataset = [] 
                    os.rename(root, root + str(index) )
                dataset.append(data)
        if(izadi):
            break
    if(izadi):
        print(index)
        break
            



