#!/usr/bin/env python 

import cPickle as pickle
import sys
import xgboost
import numpy as np
from sklearn import svm

sys.setrecursionlimit(100000)
sys.path.append("/u/lambalex/DeepLearning/undirected_matching")
sys.path.append("/u/lambalex/DeepLearning/undirected_matching/lib")

from viz import plot_images

print "loading"
from semisuper_main import discriminator
print "done loading 1"
from semisuper_main import z_to_x,x_to_z
print "done loading 2"
from utils import inverse_sigmoid
import theano
import theano.tensor as T

import time
from load_svhn import SvhnData
from load_file import normalize, denormalize
svhnData = SvhnData()

print "using UM's encoder up"
#model_name = "19080_model.pkl"
model_name = "20007_model.pkl"

param_obj = pickle.load(open("models/" + model_name, "r"))

print param_obj.keys()

dparams = param_obj['dparams']
gparams = param_obj['gparams']

input_x = T.matrix('x')

z_inf,z_feat = x_to_z(gparams, inverse_sigmoid(input_x))

d_val, d_feat = discriminator(dparams, input_x, z_inf)

#z_feat, d_feat[1]
#dfeat0, dfeat1 is what works
get_z = theano.function([input_x], outputs = [d_feat[0]])

rec = theano.function([input_x], outputs = [T.nnet.sigmoid(z_to_x(gparams,x_to_z(gparams,inverse_sigmoid(input_x))[0]))])

dhlst = []
ylst = []
t0 = time.time()

dhlst_test = []
ylst_test = []

print "all train set"
print "only using eoz features"

for ind in range(0,1000,500):

    svhn_batch = svhnData.getBatch(mb_size=500,index=ind,segment="train")
    x = normalize(svhn_batch['x']).reshape((500,3*32*32))
    y = svhn_batch['y']

    zstuff = get_z(x)

    #print zstuff[0].shape, zstuff[1].shape, zstuff[2].shape
    dhlst.append(np.concatenate(zstuff,axis=1))
    ylst.append(y)

for ind in range(0,26032-64,64):
    svhn_batch = svhnData.getBatch(mb_size=64,index=ind,segment="test")
    x = normalize(svhn_batch['x']).reshape((64,3*32*32))
    y = svhn_batch['y']
    zstuff = get_z(x)
    dhlst_test.append(np.concatenate(zstuff,axis=1))
    ylst_test.append(y)

X_train = np.vstack(dhlst)
Y_train = np.vstack(ylst).flatten()

X_test = np.vstack(dhlst_test)
Y_test = np.vstack(ylst_test).flatten()

print "label dist"
for j in range(0,9):
    print j, Y_train[Y_train==j].shape

print "starting training!"
print "using svm linear"
C = 1.0
print "C", C
print "squared hinge"

model = svm.LinearSVC(C=C,loss='squared_hinge')

model.fit(X_train, Y_train)

print "predicting held out"
y_pred = model.predict(X_test)

#print Y_test[0:100].tolist()
#print y_pred[0:100].tolist()

print "held out accuracy", np.mean(np.equal(Y_test, y_pred))

y_pred = model.predict(X_train)

print "training accuracy", np.mean(np.equal(Y_train, y_pred))

print time.time() - t0, "total time to run"




