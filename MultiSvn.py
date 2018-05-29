import tensorflow as tf
from SvnGaussiKernel import SvnGaussiKernel



class MutliSvn(object):
    def __init__(self,MAX_SEQUENCE,ALL_X,ALL_Y,BATCH_SIZE,NUM_CATEGORIES):
        self.number_of_classifiers = (NUM_CATEGORIES-1)*NUM_CATEGORIES/2
        i = j = 0
        self.all_pairs = []
        while i <self.number_of_classifiers-1:
            j = i+1
            while j <self.number_of_classifiers:
                print([i,j])
                self.all_pairs.append([i,j])
                j = j + 1
            i = i + 1
