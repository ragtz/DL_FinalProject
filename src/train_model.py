from DL_FinalProject.src.lstmModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('config', 'test', 'Model config')
tf.app.flags.DEFINE_string('data', None, 'Data numpy file')
tf.app.flags.DEFINE_string('name', None, 'Name of saved model')
tf.app.flags.DEFINE_string('losses', None, 'Losses file')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    session = tf.Session()

    print "Load input"
    lstm_input = LSTMInput(config.configs[FLAGS.config], FLAGS.data)

    print "Init Model"
    lstm_model = LSTMModel(config.configs[FLAGS.config], lstm_input, session)
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.all_variables())

    #print "Train epoch:", lstm_input.epoch_size, "batches of size", lstm_input.feature_vector_size, "x", lstm_input.num_steps
    lstm_model.train(FLAGS.losses)

    saver.save(session, FLAGS.name + '.ckpt')

    session.close()

if __name__ == '__main__':
    tf.app.run()

