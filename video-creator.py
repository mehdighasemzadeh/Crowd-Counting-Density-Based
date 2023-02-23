import cv2
import numpy as np
import glob
import os


image_folder   = 'output/image/'
density_folder = 'output/predicted/'
video_name     = 'myvideo.avi'

def create_dir(dir1):
    output = list()
    arr1 = sorted(os.listdir(dir1))
    for i in arr1:
        output.append(dir1 + i)
    return output


dir_img = create_dir(image_folder)
dir_den = create_dir(density_folder)

f = open("output/predicted/output.txt", "r")
p = list()
for x in f:
    temp = float(x)
    p.append(int(temp))
    

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1 
color = (0, 0, 255)
thickness = 2
   

   

img_array = []
for i in range(0,len(dir_img)):
    out = np.zeros((480,1280,3), np.uint8)
    img = cv2.imread(dir_img[i])
    den = cv2.imread(dir_den[i])
    den = cv2.resize(den, (640,480), interpolation = cv2.INTER_LINEAR)
    den = cv2.putText(den, str(p[i]), org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    out[:,0:640,:] = img
    out[:,640:,:]  = den 
    height, width, layers = 480, 1280, 3
    size = (width,height)
    img_array.append(out)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


