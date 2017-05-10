import os
import numpy as np

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
    return 0.0

def get_caption(text_file):
    

def getBatch():
    pass

if __name__ == "__main__":

    text_loc = '/u/lambalex/data/birds/text/text_c10/'
    image_loc = '/u/lambalex/data/birds/images/images/'

    text_files = get_text_files(text_loc)
    image_files = get_image_files(image_loc)

    keys = list(set(text_files.keys() + image_files.keys()))

    key = keys[0]

    image = get_image(text_files[key])
    caption = get_caption(image_files[key])

    print image
    print caption


