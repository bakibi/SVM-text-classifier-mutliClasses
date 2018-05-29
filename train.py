import tensorflow as tf
import numpy as np
from SvnGaussiKernel import SvnGaussiKernel
import man_data


MAX_SEQUENCE = 30
EPOCH = 500
BATCH_SIZE = 1000


def main():

    #PREPARING DATA

    training_file='data/train1.csv.zip'
    features,labels =man_data.traiter_donnees(man_data.read_file(training_file))
    features,dictionnaire = man_data.transformer_features(features,MAX_SEQUENCE)
    features,labels = man_data.labels_choose(features,labels,1,2)
    features = np.asarray(features)
    labels = np.asarray(labels)
    #features = features[:BATCH_SIZE,:]
    print(len(features),features.shape)
    #labels = labels[:BATCH_SIZE]
    labelsT = labels[:BATCH_SIZE]
    featuresT = features[:BATCH_SIZE,:]
    


    builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel')
    sess = tf.Session()
    model = SvnGaussiKernel(MAX_SEQUENCE=MAX_SEQUENCE,ALL_X=features,ALL_Y=np.transpose(labels),BATCH_SIZE=BATCH_SIZE)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_vec = []
    batch_accuracy = []
    for i in range(EPOCH):
        rand_index = np.random.choice(len(features), size=BATCH_SIZE)
        rand_x = features[rand_index]
        rand_y = np.transpose([labels[rand_index]])
        sess.run(model.train_step, feed_dict={model.x_data: rand_x, model.y_target: rand_y})

        temp_loss = sess.run(model.loss, feed_dict={model.x_data: rand_x, model.y_target: rand_y})
        loss_vec.append(temp_loss)

        acc_temp = sess.run(model.accuracy, feed_dict={model.x_data: rand_x,
                                                 model.y_target: rand_y,
                                                 model.prediction_grid:rand_x})
        batch_accuracy.append(acc_temp)

        if (i+1)%250==0:
            print('Step #' + str(i+1))
            print('Loss = ' + str(temp_loss))
            print('accuracy',acc_temp)

    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
    builder.save()

    training_file='data/small_samples1.csv'
    features,labels =man_data.traiter_donnees(man_data.read_file(training_file,False))
    features,dictionna = man_data.transformer_features(features,MAX_SEQUENCE,True,dictionnaire)
    features,labels = man_data.labels_choose(features,labels,1,2)
    features = np.asarray(features)
    labels = np.asarray(labels)

    rand_index = np.random.choice(len(features), size=BATCH_SIZE)
    test_in = features[rand_index]
    test_out = labels[rand_index]
    test_out1 = np.transpose([test_out])
    print(test_out)
    [out] = sess.run(model.prediction, feed_dict={model.x_data: featuresT,
                                             model.y_target: np.transpose([labelsT])   ,
                                             model.prediction_grid:test_in})
    print(out)

    ans = man_data.accuracy_output(test_out,out)
    print('accuary final est :',ans)



main()
