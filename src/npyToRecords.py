import tensorflow as tf
import numpy as np
import os

tf.app.flags.DEFINE_string('path', '../data', 'Directory to save tfrecords file')
tf.app.flags.DEFINE_string('filename', 'imgs', 'Name of tfrecords file')
FLAGS = tf.app.flags.FLAGS

def int64Feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convertToRecords(path, filename):
    imgs = np.load(os.path.join(path, filename + '.npy'))
    filename = os.path.join(path, filename + '.tfrecords')
    
    N = dataset.shape[0]
    rows = dataset.shape[1]
    cols = dataset.shape[2]

    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(N):
        img_raw = imgs[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={'height': int64Feature(rows),
                                                                       'width': int64Feature(cols),
                                                                       'img_raw': bytesFeature(img_raw)}))
        writer.write(example.SerializeToString())

    writer.close()

def main(argv):
    convertToRecords(FLAGS.path, FLAGS.filename)

if __name__ == '__main__':
    tf.app.run()

