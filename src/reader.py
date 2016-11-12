import tensorflow as tf
import numpy as np

def feature_vector_size(raw_data):
    _, rows, _ = raw_data.shape
    return rows

def epoch_size(raw_data, batch_size, num_steps):
    N, rows, cols = raw_data.shape
    samples_per_image = (cols - 1) / num_steps
    epoch_size = N*samples_per_image / batch_size
    return epoch_size

def batch_data_tensor(raw_data, batch_size, num_steps):
    N, rows, cols = raw_data.shape
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)

    samples_per_image = (cols - 1) / num_steps
    epoch_size = N*samples_per_image / batch_size

    assertion = tf.assert_positive(epoch_size)
    with tf.control_dependencies([assertion]):
        epoch_size = tf.identity(epoch_size, name="epoch_size")

    # get all samples
    X = raw_data[:,:,0:num_steps*samples_per_image]
    Y = raw_data[:,:,1:num_steps*samples_per_image+1]

    # flatten to 2-D
    X = tf.concat(1, [X[i,:,:] for i in range(N)])
    Y = tf.concat(1, [Y[i,:,:] for i in range(N)])

    # remove excess data
    X = tf.slice(X, [0,0], [rows, epoch_size*batch_size*num_steps])
    Y = tf.slice(Y, [0,0], [rows, epoch_size*batch_size*num_steps])

    # reshape to batches
    X = tf.reshape(tf.transpose(X), [epoch_size, batch_size, num_steps, rows])
    Y = tf.reshape(tf.transpose(Y), [epoch_size, batch_size, num_steps, rows])

    # internal transpose
    #X = tf.transpose(X, perm=[0,1,3,2])
    #Y = tf.transpose(Y, perm=[0,1,3,2])

    return X, Y

def batch_data_array(raw_data, batch_size, num_steps):
    N, rows, cols = raw_data.shape

    samples_per_image = (cols - 1) / num_steps
    epoch_size = N*samples_per_image / batch_size

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    # get all samples
    X = raw_data[:,:,0:num_steps*samples_per_image]
    Y = raw_data[:,:,1:num_steps*samples_per_image+1]

    # flatten to 2-D
    X = np.concatenate([X[i,:,:] for i in range(N)], 1)
    Y = np.concatenate([Y[i,:,:] for i in range(N)], 1)

    # remove excess data
    X = X[0:rows, 0:epoch_size*batch_size*num_steps]
    Y = Y[0:rows, 0:epoch_size*batch_size*num_steps]

    # reshape to batches
    X = np.reshape(np.transpose(X), [epoch_size, batch_size, num_steps, rows])
    Y = np.reshape(np.transpose(Y), [epoch_size, batch_size, num_steps, rows])

    return X, Y

def get_batch(X, Y, i):
    x = X[i,:,:,:]
    y = Y[i,:,:,:]
    return x, y

def img_iterator(raw_data, batch_size, num_steps, shuffle=False):
    N, rows, cols = raw_data.shape
    es = epoch_size(raw_data, batch_size, num_steps)

    X, Y = batch_data_array(raw_data, batch_size, num_steps)

    if shuffle:
        for i in np.random.permutation(es):
            x = X[i,:,:,:]
            y = Y[i,:,:,:]
            yield (x, y)
    else:
        for i in range(es):
            x = X[i,:,:,:]
            y = Y[i,:,:,:]
            yield (x, y)

def img_producer(raw_data, batch_size, num_steps, shuffle=False, name="IMGProducer"):
    with tf.name_scope(name):
        N, rows, cols = raw_data.shape
        
        X, Y = batch_data_tensor(raw_data, batch_size, num_steps)

        i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
        x = tf.squeeze(tf.slice(X, [i,0,0,0], [1, batch_size, num_steps, rows]))
        y = tf.squeeze(tf.slice(Y, [i,0,0,0], [1, batch_size, num_steps, rows]))

        return x, y

