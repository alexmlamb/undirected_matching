'''
code for loading birds dataset
'''

import os
import numpy as np
import random
from PIL import Image
from viz import plot_images
import fuel
from fuel.datasets.base import Dataset


max_caption_len = 360
image_width = 64

def normalize(x):
    return (x / 255.0).astype('float32')

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

    #print "number of images total", len(images)

    for image in images:
        if ".txt" in image:
            key = image[-14:-4]
            images_taken[key] = image

    #print "num images taken", len(images_taken)

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

    #print "number of images total", len(images)

    for image in images:
        if ".jpg" in image:
            key = image[-14:-4]
            images_taken[key] = image

    #print "num images taken", len(images_taken)

    return images_taken

def get_image(image_file):

    imgObj = Image.open(image_file).convert('RGB')

    #print np.array(imgObj).shape

    imgObj = imgObj.resize((image_width,image_width))

    img = np.asarray(imgObj).transpose(2,0,1)

    #print img.shape

    return normalize(img)

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

class BirdsData(Dataset):
    provides_sources = ('images', 'captions')

    def __init__(self,sources):

        super(BirdsData,self).__init__(sources)

        text_loc = '/u/lambalex/data/birds/text/text_c10/'
        image_loc = '/u/lambalex/data/birds/images/images/'

        self.text_files = get_text_files(text_loc)
        self.image_files = get_image_files(image_loc)

        self.keys = list(set(self.text_files.keys() + self.image_files.keys()))

    #def get_data(self, state=None, request=None):
    #    data = (numpy.random.rand(10), numpy.random.randn(3))
    #    return self.filter_sources(data)

    def get_data(self,batch_size,state=None,request=None):

        imgLst = []
        captionLst = []

        keyLst = random.sample(self.keys,50)

        for key in keyLst:
            imgLst.append([get_image(self.image_files[key])])
            captionLst.append([get_caption(self.text_files[key],index=None)])

        imageObj = np.concatenate(imgLst,axis=0)
        textObj = np.concatenate(captionLst,axis=0)

        data = (imageObj, textObj)

        return self.filter_sources(data)
    
if __name__ == "__main__":

    bd = BirdsData(sources=("images","captions")).get_data(64)

    print bd[0].shape
    print bd[1].shape

    #img,text = getBatch(keys[0:64],text_files,image_files)


    #plot_images(img.reshape((64,64*64*3)), "derp.png")




