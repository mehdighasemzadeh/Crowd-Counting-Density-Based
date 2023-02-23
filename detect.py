#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:21:04 2023

@author: mehdi
"""

# calculate number of people 
import cv2
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from datasets.crowd import Crowd
from models import M_SFANet_UCF_QNRF
import math

# Simple preprocessing.
trans = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])

# An example image 
#img = Image.open("/content/test/Dataset/001.jpg").convert('RGB')
import matplotlib.image as mpimg
import os
#%matplotlib inline 
from matplotlib import pyplot as plt
path = "test/Dataset/"
output_path = "output/predicted/"
image_output_path = "output/images/"
name_of_testdata = sorted(os.listdir(path))

model = M_SFANet_UCF_QNRF.Model()
# Weights are stored in the Google drive link.
# The model are originally trained on a GPU but, we can also test it on a CPU
model.load_state_dict(torch.load("bestmodel/Paper's_weights_UCF_QNRF/best_M-SFANet*_UCF_QNRF.pth", 
                                  map_location = torch.device('cuda')))

# Evaluation mode
model.eval()

out_put = list()
for i in range(0,201):
  img = cv2.imread(path + name_of_testdata[i])

  img = cv2.resize((img), (640,480), cv2.INTER_LINEAR)
  cv2.imwrite(image_output_path + name_of_testdata[i] , img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = trans(Image.fromarray(img))[None, :]

  density_map = model(img)
  # Est. count 
  counted = torch.sum(density_map).item()
  out_put.append(counted)
  print(" ")
  print("number of pepole in " + name_of_testdata[i] +" : " +str( counted))


  # watch density 
  np_arr = density_map.cpu().detach().numpy()
  shape = np_arr.shape
  out = np.reshape(np_arr,(shape[2],shape[3]))
  max_d = (np.amax(out))
  scale = 255/max_d
  out = out*scale
  img_density = out.astype(np.uint8)
  cv2.imwrite(output_path + name_of_testdata[i] , img_density)
  #%matplotlib inline 
  #from matplotlib import pyplot as plt
  plt.imshow(img_density, interpolation='nearest')
  plt.show()
  mpimg.imsave(output_path + name_of_testdata[i], img_density)

# ==== write predicted labels in txt file 
with open('/content/output/predicted/output.txt', 'w') as f:
    for line in out_put:
      f.write(str(line))
      f.write('\n')


