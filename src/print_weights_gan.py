from DL_FinalProject.src.lstmGanModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('config', 'test', 'Model config')
tf.app.flags.DEFINE_string('train_data', None, 'Data numpy file')
tf.app.flags.DEFINE_string('test_data', None, 'Data numpy file')
tf.app.flags.DEFINE_string('model', None, 'Name of model')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    session = tf.Session()

    print "Load input"
    lstm_input = LSTMGANInput(config.configs[FLAGS.config], FLAGS.train_data, FLAGS.test_data)

    print "Init Model"
    lstm_model = LSTMGANModel(config.configs[FLAGS.config], lstm_input, session, 'None', dtype='conv')
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.all_variables())
    saver.restore(session, FLAGS.model)

    #print [v.name for v in tf.trainable_variables()]

    conv1_W = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv1_W:0"][0].eval(sess=session)
    conv1_B = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv1_B:0"][0].eval(sess=session)

    conv2_W = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv2_W:0"][0].eval(sess=session)
    conv2_B = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv2_B:0"][0].eval(sess=session)

    conv3_W = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv3_W:0"][0].eval(sess=session)
    conv3_B = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv3_B:0"][0].eval(sess=session)

    conv4_W = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv4_W:0"][0].eval(sess=session)
    conv4_B = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/conv4_B:0"][0].eval(sess=session)

    fc1_W = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/fc1_W:0"][0].eval(sess=session)
    fc1_B = [v for v in tf.trainable_variables() if v.name == "lstm_gan_modelfc1_B:0"][0].eval(sess=session)

    fc2_W = [v for v in tf.trainable_variables() if v.name == "lstm_gan_model/fc2_W:0"][0].eval(sess=session)
    fc2_B = [v for v in tf.trainable_variables() if v.name == "lstm_gan_modelfc2_B:0"][0].eval(sess=session)

    print "conv1_W", np.min(conv1_W), np.mean(conv1_W), np.max(conv1_W)
    print "conv1_B", np.min(conv1_B), np.mean(conv1_B), np.max(conv1_B)

    print "conv2_W", np.min(conv2_W), np.mean(conv2_W), np.max(conv2_W)
    print "conv2_B", np.min(conv2_B), np.mean(conv2_B), np.max(conv2_B)

    print "conv3_W", np.min(conv3_W), np.mean(conv3_W), np.max(conv3_W)
    print "conv3_B", np.min(conv3_B), np.mean(conv3_B), np.max(conv3_B)

    print "conv4_W", np.min(conv4_W), np.mean(conv4_W), np.max(conv4_W)
    print "conv4_B", np.min(conv4_B), np.mean(conv4_B), np.max(conv4_B)

    print "fc1_W", np.min(fc1_W), np.mean(fc1_W), np.max(fc1_W)
    print "fc1_B", np.min(fc1_B), np.mean(fc1_B), np.max(fc1_B)

    print "fc2_W", np.min(fc2_W), np.mean(fc2_W), np.max(fc2_W)
    print "fc2_B", np.min(fc2_B), np.mean(fc2_B), np.max(fc2_B)

    session.close()

if __name__ == '__main__':
    tf.app.run()

