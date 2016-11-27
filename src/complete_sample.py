from DL_FinalProject.src.lstmModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('config', 'test', 'Model config')
tf.app.flags.DEFINE_string('data', None, 'Data numpy file')
tf.app.flags.DEFINE_string('model', None, 'Name of model')
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

    '''
    X = lstm_input.data[0].T
    for i in range(X.shape[0]):
        out = lstm_model.run_step([X[i,:]], False)
    '''

    X = lstm_input.data.T
    print X.shape

    '''
    samples = []
    for i in range(int(FLAGS.num_samples)):
        samples.append([np.zeros(lstm_input.feature_vector_size)])
        #samples.append([])
        for j in range(int(FLAGS.sample_length)):
            out = lstm_model.run_step([samples[i][-1]], False)
            #out = lstm_model.run_step([out], False)
            samples[i].append(out)

    np.save(FLAGS.samples, samples)
    '''
    session.close()

if __name__ == '__main__':
    tf.app.run()

