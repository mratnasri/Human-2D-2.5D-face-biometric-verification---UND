import numpy as np
import random
import os.path
import os
import cv2
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES']='0'

all_images_cropped = '../../UND 2D cropped face data/cropped_data/*.ppm'
images = []
targets = []
labels = []


def load_dataset(path, images, targets, labels):
    img_names = glob(path)
    for fn in img_names:
        img = cv2.imread(fn)
        img = cv2.resize(img, (224, 224))
        # img = load_img(fn)
        # print(img.shape)
        images.append(img)
        label = fn.split("/")[-1].split("d")[0]  # \\ for windows, / for linux
        if label not in labels:
            labels.append(label)
        target = labels.index(label)
        targets.append(target)
    return images, targets, labels


images, targets, labels = load_dataset(all_images_cropped, images, targets, labels)
images_num = len(images)
categories_n=len(labels)
print("Number of total samples = ", images_num)
print("Number of subjects = ", categories_n)
targets = np.array(targets)
idx = []
l1=0
l2=0
for i in range(categories_n):
	subject_list=list(np.where(targets == i))
	print(subject_list[0])
	idx.append(subject_list)
	if(len(subject_list[0])==1):
		l1+=1
	if(len(subject_list[0])==2):
		l2+=1

print("No. of subjects with only 1 image = ",l1)
print("No. of subjects with only 2 images = ",l2)
