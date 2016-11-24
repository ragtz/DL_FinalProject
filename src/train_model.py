from DL_FinalProject.src.lstmModel import *
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

    session = tf.Session()

    print "Load input"
    lstm_input = LSTMInput(config.configs[FLAGS.config], FLAGS.train_data, FLAGS.test_data)

    print "Init Model"
    lstm_model = LSTMModel(config.configs[FLAGS.config], lstm_input, session, FLAGS.logdir)
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

    #print "Train epoch:", lstm_input.epoch_size, "batches of size", lstm_input.feature_vector_size, "x", lstm_input.num_steps
    lstm_model.train(saver, FLAGS.model, FLAGS.save_iter, FLAGS.test_iter)

    saver.save(session, FLAGS.model + '_final.ckpt')
    #saver.save(session, FLAGS.name + '.ckpt')

    session.close()

if __name__ == '__main__':
    tf.app.run()

