from sklearn.model_selection import train_test_split
try:
    # Fix UTF8 output issues on Windows console.
    # Does nothing if package is not installed
    from win_unicode_console import enable
    enable()
except ImportError:
    pass
import tensorflow as tf
import numpy as np
import tensorflow as tf
from sklearn import datasets
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import re
from scipy import ndimage
from subprocess import check_output


MAX_SEQUENCE = 12
BATCH_SIZE = 128
NUMBER_OF_CATEGORIES = 0
###########################################################################################################
def clean_str(s):
	s = re.sub(r"[^\u0627-\u064aA-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

############################################################################################################################


# Manipuler le data
##################


values = pd.read_csv('data/train1.csv.zip', compression='zip', encoding='utf-8')
selected = ['Category', 'Descript']
non_selected = list(set(values.columns) - set(selected)) # supprimer les colonne qu on a pas besoin
values = values.drop(non_selected, axis=1)
values = values.dropna(axis=0, how='any', subset=selected)
values = values.reindex(np.random.permutation(values.index))

# extraire les categories
labels = sorted(list(set(values[selected[0]].tolist())))
num_labels = len(labels)
NUMBER_OF_CATEGORIES = num_labels
one_hot = np.zeros((num_labels, num_labels), int) #matrice carree

np.fill_diagonal(one_hot, 1)
label_dict = dict(zip(labels, one_hot))

op1 =  values[selected[0]]
labels = [label_dict[x] for x in op1]
# les description en une liste
features = values[selected[1]]

# je decompose les description en une liste de listes ou le separator est les espaces
print('decomposition')
features = features.apply(lambda x: clean_str(x).split(' ')).tolist()

#faire un groupe de mots
print('making a set')
sett = set()
llist = []
for i in range(len(features)):
    for j in range(len(features[i])):
        sett.add(features[i][j])


# faire un dictonnaire
print('faire un dictionnaire')
total = len(sett)
dict = {}
i = 1
for a in sett:
    dict[a] = (i)*1.0/total
    i = i+1
# reformer les donner
features = [[dict[x1] for x1 in x] for x in features]

new_feature = []

for i in range(len(features)):
    x = len(features[i])
    e = []
    if x <= MAX_SEQUENCE :
        e = [a  for a in features[i]]
        e = e + [0.1/total]*(MAX_SEQUENCE - x)
    else:
        e  = features[i][0:MAX_SEQUENCE]
    new_feature.append(e)

features = np.asarray(new_feature)
labels = np.asarray(labels)
labels = np.transpose(labels)
###########################################################################################################################

# preparer le model
###################

# Initialize placeholders
x_data = tf.placeholder(shape=[None, MAX_SEQUENCE], dtype=tf.float32)
y_target = tf.placeholder(shape=[NUMBER_OF_CATEGORIES, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, MAX_SEQUENCE], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[NUMBER_OF_CATEGORIES, BATCH_SIZE]))

# Gaussian (RBF) kernel
gamma = tf.constant(-100.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
#tf.constant(MAX_SEQUENCE,dtype=tf.float32)
sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))


# Declare function to do reshape/batch multiplication
def reshape_matmul(mat, _size):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [NUMBER_OF_CATEGORIES, _size, 1])
    return tf.matmul(v2, v1)

# Compute SVM Model
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target, BATCH_SIZE)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(y_target, b+1), pred_kernel)
#prediction_output = tf.matmul(b+0.5,pred_kernel)
prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []
for i in range(1000):
    rand_index = np.random.choice(len(features), size=BATCH_SIZE)
    rand_x = features[rand_index]
    rand_y = labels[:, rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)

    if (i + 1) % 25 == 0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print('Acc = '+str(acc_temp))

# Create a mesh to plot points in
x_min, x_max = features[:, 0].min() - 1, labels[:, 0].max() + 1
y_min, y_max = features[:, 1].min() - 1, labels[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]

grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
                                                   y_target: rand_y,
                                                   prediction_grid: rand_x})
print(grid_predictions)

print(grid_predictions)

#print(rand_y)
#print(grid_predictions)
#print(max(batch_accuracy),min(batch_accuracy))



####################################################################################################################


# tester le modele
##########################



values = pd.read_csv('data/small_samples1.csv', encoding='utf-8')
selected = ['Category', 'Descript']
non_selected = list(set(values.columns) - set(selected)) # supprimer les colonne qu on a pas besoin
values = values.drop(non_selected, axis=1)
values = values.dropna(axis=0, how='any', subset=selected)
values = values.reindex(np.random.permutation(values.index))


op1 =  values[selected[0]]
labels = [label_dict[x] for x in op1]


features_test = values[selected[1]]

# je decompose les description en une liste de listes ou le separator est les espaces
print('decomposition')
features_test = features_test.apply(lambda x: clean_str(x).split(' ')).tolist()


features = [[dict[x1] for x1 in x] for x in features_test]

new_feature = []

for i in range(len(features)):
    x = len(features[i])
    e = []
    if x <= MAX_SEQUENCE :
        e = [a  for a in features[i]]
        e = e + [0.1/total]*(MAX_SEQUENCE - x)
    else:
        e  = features[i][0:MAX_SEQUENCE]
    new_feature.append(e)





features_test = np.asarray(new_feature)
labels1 = np.asarray(labels)

rand_index = np.random.choice(len(features_test),size=BATCH_SIZE)
rand_xx = features_test[rand_index]
rand_xx = features_test[rand_index]
rand_yy = labels1[rand_index,:]
rand_yy = np.transpose(rand_yy)

print('PREDICTION')



prediction = sess.run(prediction,  feed_dict={x_data: rand_x,
                                                   y_target: rand_y,
                                                   prediction_grid: rand_xx})
rand_yy = np.transpose(rand_yy)
rand_yy = np.argmax(rand_yy,1)
print(rand_yy)
print(prediction)
ans = 0.0
for i in range(len(rand_yy)):
    if rand_yy[i] == prediction[i]:
        ans = ans + 1

ans = ans/len(rand_yy)
print('----------',ans)
print(max(batch_accuracy),min(batch_accuracy))
