import numpy as np
import gzip
import cPickle
import numpy.random as rng

class SvhnData:

    def __init__(self):
        np.random.seed(42)

        import scipy.io as sio
        train_file = svhn_file_train = "/u/lambalex/data/svhn/train_32x32.mat"
        extra_file = svhn_file_extra = '/u/lambalex/data/svhn/extra_32x32.mat'
        test_file = svhn_file_test = '/u/lambalex/data/svhn/test_32x32.mat'

        train_object = sio.loadmat(train_file)
        extra_object = sio.loadmat(extra_file)
        test_object = sio.loadmat(test_file)

        train_X = np.asarray(train_object["X"], dtype = 'uint8')
        extra_X = np.asarray(extra_object["X"], dtype = 'uint8')
        test_X = np.asarray(test_object["X"], dtype = 'uint8')

        train_Y = np.asarray(train_object["y"], dtype = 'uint8')
        extra_Y = np.asarray(extra_object["y"], dtype = 'uint8')
        test_Y = np.asarray(test_object["y"], dtype = 'uint8')

        train_Y -= 1
        extra_Y -= 1
        test_Y -= 1

        assert train_Y.min() == 0
        assert train_Y.max() == 9

        assert test_Y.min() == 0
        assert test_Y.max() == 9

        train_X = train_X.transpose(3,2,0,1)
        extra_X = extra_X.transpose(3,2,0,1)
        test_X = test_X.transpose(3,2,0,1)

        self.test_X = test_X

        all_train_X = np.vstack((train_X, extra_X))
        all_train_Y = np.vstack((train_Y, extra_Y))

        permutation = rng.permutation(all_train_X.shape[0])

        all_train_X = all_train_X[permutation]
        all_train_Y = all_train_Y[permutation]

        permutation = rng.permutation(extra_X.shape[0])

        self.easy_X = extra_X[permutation]
        self.easy_Y = extra_Y[permutation]

        self.train_X = all_train_X
        self.train_Y = all_train_Y

        self.test_X = test_X
        self.test_Y = test_Y

        permutation2 = rng.permutation(train_X.shape[0])

        self.hard_train_X = train_X[permutation2]
        self.hard_train_Y = train_Y[permutation2]

        self.numExamples = self.train_X.shape[0]
        print "num train examples", self.train_X.shape[0]
        print "num test examples", self.test_X.shape[0]




    def getBatch(self, index, segment, mb_size):

        #if self.index + mb_size + 10 >= self.numExamples:
        #    self.index = 0

        if segment == "train":
            mb_x = self.train_X[index : index + mb_size]
            mb_y = self.train_Y[index : index + mb_size].flatten()
        elif segment == "test":
            mb_x = self.test_X[index : index + mb_size]
            mb_y = self.test_Y[index : index + mb_size].flatten()
        elif segment == "hard_train":
            mb_x = self.hard_train_X[index : index + mb_size]
            mb_y = self.hard_train_Y[index : index + mb_size].flatten()
        elif segment == "easy_train":
            mb_x = self.easy_X[index : index + mb_size]
            mb_y = self.easy_Y[index : index + mb_size].flatten()

        return {'x' : mb_x, 'y' : mb_y}




