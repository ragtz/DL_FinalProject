from DL_FinalProject.src.lstmModel import *
import DL_FinalProject.config.config as config
import tensorflow as tf

tf.app.flags.DEFINE_string('config', 'test', 'Model config')
tf.app.flags.DEFINE_string('train_data', None, 'Data numpy file')
tf.app.flags.DEFINE_string('test_data', None, 'Data numpy file')
tf.app.flags.DEFINE_string('model', None, 'Name of model')
tf.app.flags.DEFINE_string('samples', 'sample', 'Name of saved samples file')
FLAGS = tf.app.flags.FLAGS

def main(argv):
    session = tf.Session()

    print "Load input"
    lstm_input = LSTMInput(config.configs[FLAGS.config], FLAGS.train_data, FLAGS.test_data)

    print "Init Model"
    lstm_model = LSTMModel(config.configs[FLAGS.config], lstm_input, session, 'None')
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver(tf.all_variables())
    saver.restore(session, FLAGS.model)

    '''
    X = lstm_input.data[0].T
    for i in range(X.shape[0]):
        out = lstm_model.run_step([X[i,:]], False)
    '''

    X = lstm_input.test_data
    print np.max(X)
    print np.min(X)
    samples = []

    for i in range(X.shape[0]):
        print i
        samples.append([])
        for j in range(X.shape[2]):
            if j < X.shape[2] - 50:
                out = lstm_model.run_step([X[i,:,j]], False)
                #out = np.array(255*np.clip(out, 0, 1), dtype=np.int16)/255.0
            else:
                out = lstm_model.run_step([out], False)
                #out = np.array(255*np.clip(out, 0, 1), dtype=np.int16)/255.0
                samples[i].append(out)

    np.save(FLAGS.samples, samples)

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

