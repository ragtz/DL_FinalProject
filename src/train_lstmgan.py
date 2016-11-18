from DL_FinalProject.src.lstmGanModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('config', 'test', 'Model config')
tf.app.flags.DEFINE_string('data', None, 'Data numpy file')
tf.app.flags.DEFINE_string('model', None, 'Name of saved model')
tf.app.flags.DEFINE_string('losses', None, 'Losses file')
tf.app.flags.DEFINE_string('save_iter', None, 'Save iteration')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    FLAGS.save_iter = int(FLAGS.save_iter) if FLAGS.save_iter != None else None

    session = tf.Session()

    print "Load input"
    lstmgan_input = LSTMGANInput(config.configs[FLAGS.config], FLAGS.data)

    print "Init Model"
    lstmgan_model = LSTMGANModel(config.configs[FLAGS.config], lstmgan_input, session)
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.all_variables())

    lstmgan_model.train()
    #lstmgan_model.train(saver, FLAGS.model, FLAGS.losses, FLAGS.save_iter)

    #saver.save(session, FLAGS.model + '_final.ckpt')

    session.close()

if __name__ == '__main__':
    tf.app.run()

