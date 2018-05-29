import tensorflow as tf
from SvnGaussiKernel import SvnGaussiKernel
import man_data
import numpy as np

class MutliSvm(object):
    def __init__(self,MAX_SEQUENCE,features,labels,BATCH_SIZE,NUM_CATEGORIES):
        self.number_of_classifiers = (NUM_CATEGORIES-1)*NUM_CATEGORIES/2
        i = j = 0
        self.all_pairs = []
        self.classifiers = {}
        while i <NUM_CATEGORIES-1:
            j = i+1
            while j <NUM_CATEGORIES:
                print(i,"-",j)
                x,y = man_data.labels_choose(features,labels,i,j)
                x = np.asarray(x)
                y = np.asarray(y)
                y = y[:BATCH_SIZE]
                x = x[:BATCH_SIZE,:]
                y = np.transpose([y])
                self.classifiers.update({i:{j:SvnGaussiKernel(MAX_SEQUENCE=MAX_SEQUENCE,ALL_X=x,ALL_Y=y,BATCH_SIZE=BATCH_SIZE)}})
                self.all_pairs.append([i,j])
                j = j + 1
            i = i + 1


        with tf.name_scope('training-MltiSvm'):
            self.training_result = tf.constant(45)
