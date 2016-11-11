from DL_FinalProject.src import reader
import tensorflow as tf
import numpy as np

class ImgReaderTest(tf.test.TestCase):
    def testImgProducer(self):
        raw_data = np.array([[[0,1,2,3,4],
                              [0,1,2,3,4]],

                             [[5,6,7,8,9],
                              [5,6,7,8,9]],

                             [[10,11,12,13,14],
                              [10,11,12,13,14]],

                             [[15,16,17,18,19],
                              [15,16,17,18,19]]])
        batch_size = 3
        num_steps = 2

        x, y = reader.img_producer(raw_data, batch_size, num_steps)

        with self.test_session() as session:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(session, coord=coord)

            xval, yval = session.run([x, y])
            print '--------------------'
            print xval
            print yval
            print '--------------------'
            '''
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
                coord.join()
            '''

if __name__ == '__main__':
    tf.test.main()

