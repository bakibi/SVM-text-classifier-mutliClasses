import tensorflow as tf




'''
    Cette classe decrit les differentes tranformations de la methodes gaussien
'''


class SvnGaussiKernel(object) :
    def __init__(self,MAX_SEQUENCE,ALL_X,ALL_Y,BATCH_SIZE=4):
        # Initialize placeholders
        self.x_data = tf.placeholder(shape=[None,MAX_SEQUENCE], dtype=tf.float32,name='x_data')
        self.y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32,name='y_target')
        self.prediction_grid = tf.placeholder(shape=[None,MAX_SEQUENCE], dtype=tf.float32,name='prediction_grid')
        self.all_x = tf.constant(ALL_X, dtype=tf.float32)
        self.all_y = tf.constant(ALL_Y, dtype=tf.float32)
        # Create variables for svm
        b = tf.Variable(tf.random_normal(shape=[1, BATCH_SIZE]))

        with tf.device('/cpu:0'), tf.name_scope('SVM'):
            # Gaussian (RBF) kernel
            gamma = tf.constant(-100.0,name='gamma')
            sq_dists = tf.multiply(2., tf.matmul(self.x_data, tf.transpose(self.x_data),name='mul_of_xdata__by__tr_xdata'))
            my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

            # Compute SVM Model
            first_term = tf.reduce_sum(b)
            b_vec_cross = tf.matmul(tf.transpose(b), b,name='mul_of_b__by__tr_b')
            y_target_cross = tf.matmul(self.y_target, tf.transpose(self.y_target),name='mul_of_ytarget__by__tr_ytarget')
            second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
            self.loss = tf.negative(tf.subtract(first_term, second_term))

            # Gaussian (RBF) prediction kernel
            rA = tf.reshape(tf.reduce_sum(tf.square(self.x_data), 1), [-1, 1])
            rB = tf.reshape(tf.reduce_sum(tf.square(self.prediction_grid), 1), [-1, 1])
            pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(self.x_data, tf.transpose(self.prediction_grid),name='mul_of_xdata__by__tr_predctiongrid'))), tf.transpose(rB))
            pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

            prediction_output = tf.matmul(tf.multiply(tf.transpose(self.y_target), b), pred_kernel)
            self.prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.prediction), tf.squeeze(self.y_target)), tf.float32))

            # Declare optimizer
            my_opt = tf.train.GradientDescentOptimizer(0.009)
            self.train_step = my_opt.minimize(self.loss)

            #for prediction
            rA_p = tf.reshape(tf.reduce_sum(tf.square(self.all_x), 1), [-1, 1])
            rB_p = tf.reshape(tf.reduce_sum(tf.square(self.prediction_grid), 1), [-1, 1])
            pred_sq_dist_p = tf.add(tf.subtract(rA_p, tf.multiply(2., tf.matmul(self.all_x, tf.transpose(self.prediction_grid),name='mul_of_xdata__by__tr_predctiongrid'))), tf.transpose(rB_p))
            pred_kernel_p = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist_p)))

            prediction_output_p = tf.matmul(tf.multiply(tf.transpose(self.all_y), b), pred_kernel_p)
            self.prediction_p = tf.sign(prediction_output_p - tf.reduce_mean(prediction_output_p))
