from DL_FinalProject.src.lstmModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('model', 'test', 'Model config')
tf.app.flags.DEFINE_string('data', None, 'Data numpy file')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    session = tf.Session()

    print "Load input"
    lstm_input = LSTMInput(config.configs[FLAGS.model], FLAGS.data)

    print "Init Model"
    lstm_model = LSTMModel(config.configs[FLAGS.model], lstm_input, session)

    print "Train epoch:", lstm_input.epoch_size
    lstm_model.train_epoch()

    session.close()

if __name__ == '__main__':
    tf.app.run()

