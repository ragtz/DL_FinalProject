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
                self.assertAllEqual(xval, [[[0,1],
                                            [0,1]],
 
                                           [[2,3],
                                            [2,3]],

                                           [[5,6],
                                            [5,6]]])
                self.assertAllEqual(yval, [[[1,2],
                                            [1,2]],
 
                                           [[3,4],
                                            [3,4]],

                                           [[6,7],
                                            [6,7]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[7,8],
                                            [7,8]],
 
                                           [[10,11],
                                            [10,11]],

                                           [[12,13],
                                            [12,13]]])
                self.assertAllEqual(yval, [[[8,9],
                                            [8,9]],
 
                                           [[11,12],
                                            [11,12]],

                                           [[13,14],
                                            [13,14]]])
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
                self.assertAllEqual(xval, [[[0,1],
                                            [0,1]],
 
                                           [[2,3],
                                            [2,3]]])
                self.assertAllEqual(yval, [[[1,2],
                                            [1,2]],
 
                                           [[3,4],
                                            [3,4]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[5,6],
                                            [5,6]],
 
                                           [[7,8],
                                            [7,8]]])
                self.assertAllEqual(yval, [[[6,7],
                                            [6,7]],
 
                                           [[8,9],
                                            [8,9]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[10,11],
                                            [10,11]],
 
                                           [[12,13],
                                            [12,13]]])
                self.assertAllEqual(yval, [[[11,12],
                                            [11,12]],
 
                                           [[13,14],
                                            [13,14]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[15,16],
                                            [15,16]],
 
                                           [[17,18],
                                            [17,18]]])
                self.assertAllEqual(yval, [[[16,17],
                                            [16,17]],
 
                                           [[18,19],
                                            [18,19]]])
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
                self.assertAllEqual(xval, [[[0,1,2],
                                            [0,1,2]],
 
                                           [[5,6,7],
                                            [5,6,7]]])
                self.assertAllEqual(yval, [[[1,2,3],
                                            [1,2,3]],
 
                                           [[6,7,8],
                                            [6,7,8]]])

                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[[10,11,12],
                                            [10,11,12]],
 
                                           [[15,16,17],
                                            [15,16,17]]])
                self.assertAllEqual(yval, [[[11,12,13],
                                            [11,12,13]],
 
                                           [[16,17,18],
                                            [16,17,18]]])
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
                self.assertAllEqual(xval, [[[0,1],
                                            [0,1]],
 
                                           [[2,3],
                                            [2,3]],

                                           [[5,6],
                                            [5,6]]

                                           [[7,8],
                                            [7,8]],

                                           [[10,11],
                                            [10,11]]])
                self.assertAllEqual(yval, [[[1,2],
                                            [1,2]],
 
                                           [[3,4],
                                            [3,4]],

                                           [[6,7],
                                            [6,7]]

                                           [[8,9],
                                            [8,9]],

                                           [[11,12],
                                            [11,12]]])
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
                self.assertAllEqual(xval, [[[0,1,2,3],
                                            [0,1,2,3]],
 
                                           [[5,6,7,8],
                                            [5,6,7,8]],

                                           [[10,11,12,13],
                                            [10,11,12,13]]])
                self.assertAllEqual(yval, [[[1,2,3,4],
                                            [1,2,3,4]],
 
                                           [[6,7,8,9],
                                            [6,7,8,9]],

                                           [[11,12,13,14],
                                            [11,12,13,14]]])
            finally:
                coord.request_stop()
                coord.join(threads)    

if __name__ == '__main__':
    tf.test.main()

