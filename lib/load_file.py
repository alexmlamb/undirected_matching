import glob
from PIL import Image
import numpy as np
from viz import plot_images

def normalize(x):
    return (x / 255.0)

def denormalize(x):
    return (x) * 255.0

class FileData:


    def __init__(self, loc, width, mb):

        self.lastIndex = 0

        if loc[-1] != "/":
            loc += "/"

        images = glob.glob(loc + "*")

        for image in images:
            assert "jpg" in image or "png" in image

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

    
    loc = "/u/lambalex/DeepLearning/animefaces/faces/danbooru-faces/uniform/"

    imageNetData = FileData(loc, 100, 36)

    for i in range(0,1):
        x = imageNetData.getBatch()
        #print (x - imageNetData.denormalize(imageNetData.normalize(x))).mean()
        print normalize(x).max()
        print normalize(x).min()

        print x.shape

        plot_images(normalize(x), "derp.png")



