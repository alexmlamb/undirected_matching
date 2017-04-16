import numpy as np
import gzip
import cPickle


class SvhnData:

    def __init__(self, segment, config):
        np.random.seed(config["seed"])

        import scipy.io as sio
        train_file = config["svhn_file_train"]
        extra_file = config["svhn_file_extra"]
        test_file = config["svhn_file_test"]

        train_object = sio.loadmat(train_file)
        extra_object = sio.loadmat(extra_file)
        test_object = sio.loadmat(test_file)

        train_X = np.asarray(train_object["X"], dtype = 'float32')
        extra_X = np.asarray(extra_object["X"], dtype = 'float32')
        test_X = np.asarray(test_object["X"], dtype = 'float32')

        train_Y = np.asarray(train_object["y"], dtype = 'uint8')
        extra_Y = np.asarray(extra_object["y"], dtype = 'uint8')
        test_Y = np.asarray(test_object["y"], dtype = 'uint8')

        train_Y -= 1
        extra_Y -= 1
        test_Y -= 1

        assert train_Y.min() == 0
        assert train_Y.max() == 9

        train_X = np.swapaxes(np.swapaxes(np.swapaxes(train_X, 0,3), 2,3), 1,2)

        extra_X = np.swapaxes(np.swapaxes(np.swapaxes(extra_X, 0,3), 2, 3), 1, 2)

        test_X = np.swapaxes(np.swapaxes(np.swapaxes(test_X, 0,3), 2, 3), 1, 2)

        self.test_X = test_X

        train_X = np.vstack((train_X, extra_X))
        train_Y = np.vstack((train_Y, extra_Y))

        old_seed = np.random.randint(low=0, high=np.iinfo(np.uint32).max)
        np.random.seed(42)
        train_indices = np.random.choice(train_X.shape[0], int(train_X.shape[0] * (1.0 - config["fraction_validation"])), replace = False)
        valid_indices = np.setdiff1d(range(0,train_X.shape[0]), train_indices)

        self.mb_size = config['mb_size']

        self.valid_X = train_X[valid_indices]
        self.valid_Y = train_Y[valid_indices]

        self.train_X = train_X[train_indices]
        self.train_Y = train_Y[train_indices]

        if segment == "train":
            self.dataobj = self.train_X
        elif segment == "test":
            self.dataobj = self.test_X

        self.numExamples = self.dataobj.shape[0]

        self.index = 0

        print "shape", self.train_X.shape

        np.random.seed(old_seed)

    def normalize(self, x):
        return (x / 127.5) - 1.0

    def denormalize(self, x):
        return (x + 1.0) * 127.5


    def getBatch(self):

        if self.index + self.mb_size + 10 >= self.numExamples:
            self.index = 0

        mb = self.dataobj[self.index : self.index + self.mb_size]

        self.index += self.mb_size

        return {'x' : mb, 'labels' : np.zeros(self.mb_size).astype('int32')}


