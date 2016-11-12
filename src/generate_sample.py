from DL_FinalProject.src.lstmModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('config', 'test', 'Model config')
tf.app.flags.DEFINE_string('model', None, 'Name of model')
tf.app.flags.DEFINE_string('num_samples', 1, 'Number of samples to generate')
tf.app.flags.DEFINE_string('sample_length', 500, 'Length of generated samples')
tf.app.flags.DEFINE_string('samples', 'sample', 'Name of saved samples file')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    session = tf.Session()

    print "Load input"
    lstm_input = LSTMInput(config.configs[FLAGS.config], FLAGS.data)

    print "Init Model"
    lstm_model = LSTMModel(config.configs[FLAGS.config], lstm_input, session)
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.all_variables())
    saver.restore(session, FLAGS.model)

    samples = []
    for i in range(FLAGS.num_samples):
        samples.append([np.zeros(FLAGS.config.feature_vector_size)])
        for j in range(FLAGS.sample_length):
            out = lstm_model.run_step(samples[i][-1], False)
            samples[i].append(out)

    np.save(FLAGS.samples)

    session.close()

if __name__ == '__main__':
    tf.app.run()

