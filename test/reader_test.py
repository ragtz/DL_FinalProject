from DL_FinalProject.src import reader
import tensorflow as tf
import numpy as np

class ImgReaderTest(tf.test.TestCase):
    def setUp(self):
        self.raw_data = np.array([[[0,1,2,3,4],
                                   [0,1,2,3,4]],

                                  [[5,6,7,8,9],
                                   [5,6,7,8,9]],

                                  [[10,11,12,13,14],
                                   [10,11,12,13,14]],

                                  [[15,16,17,18,19],
                                   [15,16,17,18,19]]])

    def testImgProducer0(self):
        batch_size = 3
        num_steps = 2

        x, y = reader.img_producer(self.raw_data, batch_size, num_steps)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord=coord)
            
            try:
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[0,0],
                                            [1,1]],
 
                                           [[2,2],
                                            [3,3]],

                                           [[5,5],
                                            [6,6]]])
                self.assertAllEqual(yval, [[[1,1],
                                            [2,2]],
 
                                           [[3,3],
                                            [4,4]],

                                           [[6,6],
                                            [7,7]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[7,7],
                                            [8,8]],
 
                                           [[10,10],
                                            [11,11]],

                                           [[12,12],
                                            [13,13]]])
                self.assertAllEqual(yval, [[[8,8],
                                            [9,9]],
 
                                           [[11,11],
                                            [12,12]],

                                           [[13,13],
                                            [14,14]]])
            finally:
                coord.request_stop()
                coord.join(threads)

    def testImgProducer1(self):
        batch_size = 2
        num_steps = 2

        x, y = reader.img_producer(self.raw_data, batch_size, num_steps)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord=coord)
            
            try:
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[0,0],
                                            [1,1]],
 
                                           [[2,2],
                                            [3,3]]])
                self.assertAllEqual(yval, [[[1,1],
                                            [2,2]],
 
                                           [[3,3],
                                            [4,4]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[5,5],
                                            [6,6]],
 
                                           [[7,7],
                                            [8,8]]])
                self.assertAllEqual(yval, [[[6,6],
                                            [7,7]],
 
                                           [[8,8],
                                            [9,9]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[10,10],
                                            [11,11]],
 
                                           [[12,12],
                                            [13,13]]])
                self.assertAllEqual(yval, [[[11,11],
                                            [12,12]],
 
                                           [[13,13],
                                            [14,14]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[15,15],
                                            [16,16]],
 
                                           [[17,17],
                                            [18,18]]])
                self.assertAllEqual(yval, [[[16,16],
                                            [17,17]],
 
                                           [[18,18],
                                            [19,19]]])
            finally:
                coord.request_stop()
                coord.join(threads)

    def testImgProducer2(self):
        batch_size = 2
        num_steps = 3

        x, y = reader.img_producer(self.raw_data, batch_size, num_steps)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord=coord)
            
            try:
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[0,0],
                                            [1,1],
                                            [2,2]],
 
                                           [[5,5],
                                            [6,6],
                                            [7,7]]])
                self.assertAllEqual(yval, [[[1,1],
                                            [2,2],
                                            [3,3]],
 
                                           [[6,6],
                                            [7,7],
                                            [8,8]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[10,10],
                                            [11,11],
                                            [12,12]],
 
                                           [[15,15],
                                            [16,16],
                                            [17,17]]])
                self.assertAllEqual(yval, [[[11,11],
                                            [12,12],
                                            [13,13]],
 
                                           [[16,16],
                                            [17,17],
                                            [18,18]]])
            finally:
                coord.request_stop()
                coord.join(threads)

    def testImgProducer3(self):
        batch_size = 5
        num_steps = 2

        x, y = reader.img_producer(self.raw_data, batch_size, num_steps)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord=coord)
            
            try:
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[0,0],
                                            [1,1]],
 
                                           [[2,2],
                                            [3,3]],

                                           [[5,5],
                                            [6,6]],

                                           [[7,7],
                                            [8,8]],

                                           [[10,10],
                                            [11,11]]])
                self.assertAllEqual(yval, [[[1,1],
                                            [2,2]],
 
                                           [[3,3],
                                            [4,4]],

                                           [[6,6],
                                            [7,7]],

                                           [[8,8],
                                            [9,9]],

                                           [[11,11],
                                            [12,12]]])
            finally:
                coord.request_stop()
                coord.join(threads)

    def testImgProducer4(self):
        batch_size = 3
        num_steps = 4

        x, y = reader.img_producer(self.raw_data, batch_size, num_steps)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord=coord)
            
            try:
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[0,0],
                                            [1,1],
                                            [2,2],
                                            [3,3]],
 
                                           [[5,5],
                                            [6,6],
                                            [7,7],
                                            [8,8]],

                                           [[10,10],
                                            [11,11],
                                            [12,12],
                                            [13,13]]])
                self.assertAllEqual(yval, [[[1,1],
                                            [2,2],
                                            [3,3],
                                            [4,4]],
 
                                           [[6,6],
                                            [7,7],
                                            [8,8],
                                            [9,9]],

                                           [[11,11],
                                            [12,12],
                                            [13,13],
                                            [14,14]]])
            finally:
                coord.request_stop()
                coord.join(threads)

    def testRealImg(self):
        data = np.load('../data/spectrograms/spectrograms.npy')
        batch_size = 256
        num_steps = 32
        epoch_size = reader.epoch_size(data, batch_size, num_steps)

        x, y = reader.img_producer(data, batch_size, num_steps)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(session, coord=coord)

            try:
                print epoch_size
                for i in range(epoch_size):
                    xval, yval = session.run([x, y])
                    print "Batch", i

            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    tf.test.main()

