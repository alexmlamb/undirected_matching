
#!/usr/bin/env python

# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")

import theano
import theano.tensor as T
from load_cifar import CifarData
import time

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
#import scipy.misc
import math

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
#was 10 splits
def get_inception_score(images, splits=2):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 500
  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        qw = np.asarray(inp)
        qw = qw.reshape((bs,32,32,3))
        inp = qw.tolist()
        #inp = np.concatenate(inp, 0)
        t0 = time.time()
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        print('time in 1 mb')
        print(time.time()-t0)
        print(pred.shape)
        #print(pred.tolist())
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    print("preds shape")
    print(preds.shape)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))

    print("scores length")
    print(len(scores))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()


data_file = sys.argv[1]#"saved_data/12480_inpainting.bin"

data = np.load(open(data_file, "r"))['arr_0']

data = data.reshape((data.shape[0],3,32,32)).transpose(0,2,3,1)*255.0

print("Shape before pruning")
print(data.shape)
print(data.min())
print(data.max())

data_lst = []

for i in range(0, data.shape[0] - data.shape[0]%500):
    data_lst.append(data[i])

data_lst = data_lst[100000:150000]

print(len(data_lst))

a,b = get_inception_score(data_lst)

print(a)
print(b)

raise Exception('done')

#a, b = get_inception_score(images)
#print(a)
#print(b)







config = {}

config["cifar_location"] = "/u/lambalex/data/cifar/cifar-10-batches-py/"
config['mb_size'] = 50000
config['image_width'] = 32

cd = CifarData(config, segment = "train")

real_batch = cd.getBatch()['x']

#from chainer import serializers, datasets
#train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)

#real_batch = train[:1000]

batch_lst = []

for i in range(0,real_batch.shape[0]):
    mod = np.concatenate([real_batch[i,:,:16,:],real_batch[real_batch.shape[0] - i - 1,:,16:,:]],axis=1)
    print("mod shape")
    print(mod.shape)
    real_batch[i,:,:16,:] *= 0.0
    #batch_lst.append(mod.transpose(1,2,0))#real_batch[i].transpose(1,2,0))
    batch_lst.append(real_batch[i].transpose(1,2,0))

a,b = get_inception_score(batch_lst)

print("real incept")
print(a)
print(b)



