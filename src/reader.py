import tensorflow as tf
import numpy as np

def img_producer(raw_data, batch_size, num_steps, shuffle=False, name="IMGProducer"):
    with tf.name_scope(name):
        N, rows, cols = raw_data.shape
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        samples_per_image = (cols - 1) / num_steps
        epoch_size = N*samples_per_image/batch_size

        assertion = tf.assert_positive(epoch_size)
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        # get all samples and flatten to 2-D
        #X = tf.reshape(raw_data[:,:,0:num_steps*samples_per_image], [rows, N*num_steps*samples_per_image])
        #Y = tf.reshape(raw_data[:,:,1:num_steps*samples_per_image+1], [rows, N*num_steps*samples_per_image])
        #X = tf.reshape(raw_data[:,:,0:num_steps*samples_per_image], [N*rows, num_steps*samples_per_image])
        #Y = tf.reshape(raw_data[:,:,1:num_steps*samples_per_image+1], [N*rows, num_steps*samples_per_image])
        X = tf.concat(0, raw_data[:,:,0:num_steps*samples_per_image])
        Y = tf.concat(0, raw_data[:,:,1:num_steps*samples_per_image+1])

        # remove excess data
        #X = tf.slice(X, [0,0], [rows, epoch_size*batch_size*num_steps])
        #Y = tf.slice(Y, [0,0], [rows, epoch_size*batch_size*num_steps])

        # reshape to batches
        #X = tf.reshape(X, [epoch_size, batch_size, rows, num_steps])
        #Y = tf.reshape(Y, [epoch_size, batch_size, rows, num_steps])

        #X = tf.reshape(raw_data[:,:,0:num_steps*samples_per_image], [epoch_size, batch_size, rows, num_steps])
        #Y = tf.reshape(raw_data[:,:,1:num_steps*samples_per_image+1], [epoch_size, batch_size, rows, num_steps])        

        #i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
        #x = tf.slice(X, [i,0,0,0], [1, batch_size, rows, num_steps])
        #y = tf.slice(Y, [i,0,0,0], [1, batch_size, rows, num_steps])

        return X, Y #x, y

