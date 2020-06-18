#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv2
import pandas as pd
import os
from skimage import io,transform
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

print(cv2.__version__)
df = pd.read_csv('D:/Zavrsni/mycsv.csv') #učitavanje pozicije okvira
classes = df.class_name.unique().tolist()
IMAGES_PATH = 'D:/Zavrsni/faces' #path do dataseta
unique_files = df.file_name.unique()
classifier = cv2.CascadeClassifier('D:/zavrsni/haarcascade_frontalface_default.xml') #učitati trenirani haarcascade model

# In[3]:


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

      poly = [
          (xmin, ymin), (xmax, ymin),
          (xmax, ymax), (xmin, ymax)
      ]
      poly = list(itertools.chain.from_iterable(poly))

      obj = [xmin, ymin, xmax, ymax]
      
      objs = objs + [obj]

    record["boxes"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

dataset = create_dataset_dicts(df,classes)


# In[5]:


def mysort(val): 
    return val[0] 

brojac = 0
tocni = 0
totalni_pronadeni = 0
totalni_rezultati = 0
detected_boxes = []
print(len(dataset[:]))


# In[ ]:


for data in dataset[:]:
    detected_boxes = []
    data['boxes'].sort(key=mysort)
    brojac +=1
    pixels = cv2.imread(data['file_name'])
    bboxes = classifier.detectMultiScale(pixels)
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
            xmin,ymin,xmax,ymax = box
            pxmin,pymin,width,height = detected_boxes[j]
            
            pxmax = pxmin + width
            pymax = pymin + height
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
            if(iou == 0):

    for m in range(0,len(max_iou)):
        if (max_iou[m] > 0.5):
            tocni +=1 #presjek od 30% je dovoljan za prihvaćanje rezultata pošto su rezultati sami po sebi dosta široki ili pomaknut

    if brojac >= 13646: #zadnja slika je 13646
        print("correct predictions, total predicitons, total truths")
        print(tocni,totalni_pronadeni,totalni_rezultati)
        break
    

