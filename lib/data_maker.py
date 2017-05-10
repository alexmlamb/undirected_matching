'''
code for loading birds dataset
'''

import os
import numpy as np
import random
from PIL import Image

max_caption_len = 360
image_width = 64

def caption2arr(caption):
    arr = np.zeros(shape = (max_caption_len,))

    for charInd in range(len(caption)):
        char = caption[charInd]
        arr[charInd] = ord(char)

    return arr

def get_text_files(loc):
    if loc[-1] != "/":
        loc += "/"

    images = []
    dirs = os.walk(loc)

    for dirp in dirs:
        for fi in dirp[2]:
            images.append(dirp[0] + "/" + fi)

    images_taken = {}

    print "number of images total", len(images)

    for image in images:
        if ".txt" in image:
            key = image[-14:-4]
            images_taken[key] = image

    print "num images taken", len(images_taken)

    return images_taken

def get_image_files(loc):
    if loc[-1] != "/":
        loc += "/"

    images = []
    dirs = os.walk(loc)

    for dirp in dirs:
        for fi in dirp[2]:
            images.append(dirp[0] + "/" + fi)

    images_taken = {}

    print "number of images total", len(images)

    for image in images:
        if ".jpg" in image:
            key = image[-14:-4]
            images_taken[key] = image

    print "num images taken", len(images_taken)

    return images_taken

def get_image(image_file):

    imgObj = Image.open(image_file).convert('RGB')

    imgObj = imgObj.resize((image_width,image_width))
    img = np.asarray(imgObj)

    return img

def get_caption(text_file,index):

    lines = open(text_file,"r")
    linelst = []

    for line in lines:
        linelst.append(line)

    if index == None:
        index = random.randint(0,len(linelst)-1)
    
    line = linelst[index]

    arr = caption2arr(line)

    return arr

def getBatch(keyLst,text_files,image_files):

    imgLst = []
    captionLst = []

    for key in keyLst:
        imgLst.append([get_image(image_files[key])])
        captionLst.append([get_caption(text_files[key],index=None)])

    imageObj = np.concatenate(imgLst,axis=0)
    textObj = np.concatenate(captionLst,axis=0)

    return imageObj, textObj

if __name__ == "__main__":

    text_loc = '/u/lambalex/data/birds/text/text_c10/'
    image_loc = '/u/lambalex/data/birds/images/images/'

    text_files = get_text_files(text_loc)
    image_files = get_image_files(image_loc)

    keys = list(set(text_files.keys() + image_files.keys()))

    img,text = getBatch(keys[0:10],text_files,image_files)

    print img.shape
    print text.shape


