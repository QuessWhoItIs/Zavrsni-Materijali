#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2, random
import face_detection
from utils import draw_bboxes, draw_bbox
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time


# In[2]:


detector = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)
# BGR to RGB
#im = cv2.imread("D:/zavrsni/faces/face_332.jpeg")[:, :, ::-1]


# In[3]:


df = pd.read_csv('D:/Zavrsni/mycsv.csv')
classes = df.class_name.unique().tolist()
IMAGES_PATH = 'D:/Zavrsni/faces'
unique_files = df.file_name.unique()


# In[4]:


def create_dataset_dicts(df, classes):
  dataset_dicts = []
  for image_id, img_name in enumerate(df.file_name.unique()):

    record = {}

    image_df = df[df.file_name == img_name]

    file_path = f'{IMAGES_PATH}/{img_name}'
    record["file_name"] = file_path
    record["image_id"] = image_id
    record["height"] = int(image_df.iloc[0].height)
    record["width"] = int(image_df.iloc[0].width)

    objs = []
    for _, row in image_df.iterrows():

      xmin = int(row.x_min)
      ymin = int(row.y_min)
      xmax = int(row.x_max)
      ymax = int(row.y_max)

      obj = [xmin, ymin, xmax, ymax]
      
      objs = objs + [obj]

    record["boxes"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

dataset = create_dataset_dicts(df,classes)


# In[5]:


def draw_image_with_boxes(filename, result_list, predict_list):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    # get coordinates
    x = result_list[0]
    y = result_list[1]
    width = result_list[2]
    height = result_list[3]
    
    x2 = predict_list[0]
    y2 = predict_list[1]
    width2 = predict_list[2]
    height2 = predict_list[3]
    # create the shape
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    rect2 = Rectangle((x2, y2), width2, height2, fill=False, color='blue')
    # draw the box
    ax.add_patch(rect)
    ax.add_patch(rect2)
    # show the plot
    plt.show()


# In[6]:


def mysort(val): 
    return val[0] 

brojac = 0
tocni = 0
totalni_pronadeni = 0
totalni_rezultati = 0
detected_boxes = []
#print(len(dataset[:]))


# In[7]:


start = time.time()
for data in dataset[:]:
    detected_boxes = []
    data['boxes'].sort(key=mysort)
    brojac +=1
    pixels = cv2.imread(data['file_name'])[:, :, ::-1]
    original_width = pixels.shape[1]
    original_height = pixels.shape[0]
    pixels = cv2.resize(pixels,(300,300))
    bboxes = detector.detect(pixels)
    for box in bboxes:
        box = box.tolist()
        detected_boxes = detected_boxes + [box] #dodaj svako pronadeno lice na jedno mjesto u obliku liste
    detected_boxes.sort(key=mysort)
    temp = []
    max_iou = [0]*len(data['boxes'])

    totalni_pronadeni += len(detected_boxes)
    for index in range(0,len(data['boxes'])): #za svako riješenje pokušaj ga spojiti sa pronađenim licem
        totalni_rezultati += 1
        boxes = data['boxes']
        box = boxes[index]
        for j in range(0 , len(detected_boxes)):
            xmin,ymin,xmax,ymax = box[:]
            pxmin,pymin,pxmax,pymax = detected_boxes[j][:-1]

            pxmin = pxmin*original_width/300 #magični brojevi predstavljaju širinu ili visinu ovisno o koordinati
            pxmax = pxmax*original_width/300
            pymin = pymin*original_height/300
            pymax = pymax*original_height/300
            
            interxmin = max(xmin,pxmin)
            interymin = max(ymin,pymin)
            interxmax = min(xmax,pxmax)
            interymax = min(ymax,pymax)
            
            inter_area = max(0, interxmax - interxmin + 1) * max(0, interymax - interymin + 1)
            actual_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            pred_area = (pxmax - pxmin + 1) * (pymax - pymin + 1)
            iou = inter_area / float(actual_area + pred_area - inter_area)
            if(iou>max_iou[index]):
                max_iou[index] = iou

    for m in range(0,len(max_iou)):
        if (max_iou[m] > 0.2):
            tocni +=1 #presjek od 20% je dovoljan za prihvaćanje rezultata 
            #zato što je model vrlo točan ali prikazuje manje okvire zato što je treniran na WIDER FACE skupu.

    if brojac >= 13646:
        print("correct predictions, total predicitons, total truths")
        print(tocni,totalni_pronadeni,totalni_rezultati)
        end = time.time()
        print(end-start)
        break
 