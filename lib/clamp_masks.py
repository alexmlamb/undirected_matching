#generates masks for clamping
import numpy as np

def left_half_missing():

    mask = np.ones(shape = (64,3,32,32)).astype('float32')

    mask[:,:,:16,:] *= 0.0

    return mask.reshape((64,3*32*32))

def block_occ():

    return mask

if __name__ == "__main__":

    print left_half_missing()[0][0].tolist()

