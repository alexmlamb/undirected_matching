import glob
from PIL import Image
import numpy as np
from viz import plot_images
import os


def normalize(x):
    return (x / 255.0)

def denormalize(x):
    return (x) * 255.0

class FileData:


    def __init__(self, loc, width, mb):

        self.lastIndex = 0

        if loc[-1] != "/":
            loc += "/"

        #images = glob.glob(loc + "**")

        images = []
        dirs = os.walk(loc)

        for dirp in dirs:
            for fi in dirp[2]:
                images.append(dirp[0] + "/" + fi)

        images_taken = []

        print "number of images total", len(images)

        for image in images:
            if "jpg" in image or "png" in image:
                images_taken.append(image)

        images = images_taken

        print "number of images taken", len(images_taken)
        self.numExamples = len(images)
        self.images = images

        self.mb_size = mb
        self.image_width = width



    def getBatch(self):
        
        imageLst = []

        index = self.lastIndex

        while len(imageLst) < self.mb_size:
            image = self.images[index]
            try:
                imgObj = Image.open(image).convert('RGB')
            except:
                continue

            imgObj = imgObj.resize((self.image_width,self.image_width))
            img = np.asarray(imgObj)
            if img.shape == (self.image_width,self.image_width,3):
                imageLst.append([img])

            index += 1
            if index >= self.numExamples:
                index = 0


            imgObj.close()

        x = np.vstack(imageLst).astype('float32')

        x = x.transpose(0,3,1,2)

        self.lastIndex = index + 1

        return x

if __name__ == "__main__":

    
    loc = "/u/lambalex/DeepLearning/animefaces/datafaces/danbooru-faces/"

    imageNetData = FileData(loc, 64, 64)

    print "loaded"

    for i in range(0,1):
        x = imageNetData.getBatch()
        #print (x - imageNetData.denormalize(imageNetData.normalize(x))).mean()
        print normalize(x).max()
        print normalize(x).min()

        print x.shape

        plot_images(normalize(x).reshape((64,64*64*3)), "derp.png")

