from DL_FinalProject.src.lstmGanModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('config', 'test', 'Model config')
tf.app.flags.DEFINE_string('train_data', None, 'Train data numpy file')
tf.app.flags.DEFINE_string('test_data', None, 'Test data numpy file')
tf.app.flags.DEFINE_string('model', None, 'Name of saved model')
tf.app.flags.DEFINE_string('logdir', None, 'Summary log directory')
tf.app.flags.DEFINE_string('save_iter', None, 'Save iteration')
tf.app.flags.DEFINE_string('test_iter', None, 'Test iteration')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    FLAGS.save_iter = int(FLAGS.save_iter) if FLAGS.save_iter != None else None
    FLAGS.test_iter = int(FLAGS.test_iter) if FLAGS.test_iter != None else None

    session = tf.Session()

    print "Load input"
    lstmgan_input = LSTMGANInput(config.configs[FLAGS.config], FLAGS.train_data, FLAGS.test_data)

    print "Init Model"
    lstmgan_model = LSTMGANModel(config.configs[FLAGS.config], lstmgan_input, session, FLAGS.logdir)
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

    lstmgan_model.train(FLAGS.test_iter)
    #lstmgan_model.train(saver, FLAGS.model, FLAGS.losses, FLAGS.save_iter)

    #saver.save(session, FLAGS.model + '_final.ckpt')

    session.close()

if __name__ == '__main__':
    tf.app.run()

