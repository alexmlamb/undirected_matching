'''
code for loading birds dataset
'''

import os
import random

import cv2
import fuel
from fuel.datasets.base import Dataset
from fuel.datasets.hdf5 import H5PYDataset
import h5py
import numpy as np
from PIL import Image

from viz import plot_images

max_caption_len = 360
image_width = 64
text_loc = '/u/lambalex/data/birds/text/text_c10/'
image_loc = '/u/lambalex/data/birds/images/images/'


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

    arr = np.asarray(imgObj)
    arr = cv2.resize(arr, (image_width, image_width),
                     interpolation=cv2.INTER_AREA)
    img = arr.transpose(2, 0, 1)
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

class BirdsData(Dataset):
    provides_sources = ('images', 'captions')

    def __init__(self, sources):

        super(BirdsData,self).__init__(sources)

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

    
def make_hdf5(file_path):
    f = h5py.File(file_path, mode='w')
    text_files = get_text_files(text_loc)
    image_files = get_image_files(image_loc)
    captions = np.array(
        [np.array([get_caption(
            text_files[text_files.keys()[j]], i) for i in xrange(10)])
         for j in xrange(len(text_files))])

    chr_map = dict((int(c), i) for i, c in enumerate(
        np.unique(captions).tolist()))
    c_map = dict((v, chr(k)) for k, v in chr_map.items())
    captions = f.create_dataset(
        'captions', (len(text_files), 10, 360), dtype='uint8')
    images = f.create_dataset(
        'features', (len(image_files), 3, image_width, image_width),
        dtype='float32')
    for j, k in enumerate(text_files.keys()):
        cap = np.array([map(lambda x: chr_map[x],
                            get_caption(text_files[k], i))
                        for i in xrange(10)])
        captions[j, ...] = cap
        
        image = get_image(image_files[k])
        images[j, ...] = image
    split_dict = {'train': {'features': (0, 10788),
                            'captions': (0, 10788)},
                  'test': {'features': (10788, 11788),
                           'captions': (10788, 11788)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.attrs['char_map'] = [(int(k), v) for k, v in c_map.items()]
    
    f.flush()
    f.close()
        
if __name__ == "__main__":

    bd = BirdsData(sources=("images","captions")).get_data(64)

    print bd[0].shape
    print bd[1].shape

    #img,text = getBatch(keys[0:64],text_files,image_files)


    #plot_images(img.reshape((64,64*64*3)), "derp.png")




