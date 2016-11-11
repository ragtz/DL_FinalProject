import tensorflow as tf
import numpy as np

def img_producer(raw_data, batch_size, num_steps, shuffle=False, name="IMGProducer"):
    with tf.name_scope(name):
        N, rows, cols = raw_data.shape
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        samples_per_image = (cols - 1) / num_steps
        epoch_size = N*samples_per_image/batch_size
        print epoch_size

        assertion = tf.assert_positive(epoch_size)
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        X = tf.reshape(raw_data[:,:,0:num_steps*samples_per_image], [epoch_size, batch_size, rows, num_steps])
        Y = tf.reshape(raw_data[:,:,1:num_steps*samples_per_image+1], [epoch_size, batch_size, rows, num_steps])        

        i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
        x = X[i,:,:,:]
        y = Y[i,:,:,:]

        return x, y

