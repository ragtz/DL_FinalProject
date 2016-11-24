from DL_FinalProject.src import reader
import tensorflow as tf
import numpy as np

class LSTMInput(object):
    def __init__(self, config, train_data_path, test_data_path, name=None):
        self.data = np.load(train_data_path)
        self.test_data = np.load(test_data_path)
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.test_num_steps = config.test_num_steps
        self.feature_vector_size = reader.feature_vector_size(self.data)
        self.epoch_size = reader.epoch_size(self.data, self.batch_size, self.num_steps)
        self.X, self.Y = reader.batch_data_array(self.data, self.batch_size, self.num_steps)
        self.test_X, self.test_Y = reader.get_test_arrays(self.test_data, self.test_num_steps)
        self.test_size = self.test_X.shape[0]

class LSTMModel(object):
    def __init__(self, config, lstm_input, session, summary_dir, name="lstm_model"):
        self.scope = name
        self.session = session
        self.summary_dir = summary_dir

        self.config = config
        self.lstm_input = lstm_input

        self.lstm_last_state = np.zeros((2*self.config.num_layers*self.config.hidden_size,))
        #self.lstm_last_state = tf.placeholder(tf.float32, shape=(None, 2*self.config.num_layers*self.config.hidden_size), name="last_state")

        with tf.variable_scope(self.scope):
            self.xbatch = tf.placeholder(tf.float32, shape=(None, None, self.lstm_input.feature_vector_size), name="xbatch")
            self.initial_state = tf.placeholder(tf.float32, shape=(None, 2*self.config.num_layers*self.config.hidden_size), name="initial_state")

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0, state_is_tuple=False)
            self.lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers)

            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xbatch, initial_state=self.initial_state)

            self.rnn_out_W = tf.Variable(tf.random_normal((self.config.hidden_size, self.lstm_input.feature_vector_size), stddev=0.01))
            self.rnn_out_B = tf.Variable(tf.random_normal((self.lstm_input.feature_vector_size,), stddev=0.01))

            outputs_reshaped = tf.reshape(outputs, [-1, self.config.hidden_size])
            network_output = (tf.matmul(outputs_reshaped, self.rnn_out_W) + self.rnn_out_B)

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(network_output, (batch_time_shape[0], batch_time_shape[1], self.lstm_input.feature_vector_size))

            self.ybatch = tf.placeholder(tf.float32, (None, None, self.lstm_input.feature_vector_size))
            ybatch_reshaped = tf.reshape(self.ybatch, [-1, self.lstm_input.feature_vector_size])

            self.loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(network_output, ybatch_reshaped)))
            self.train_op = tf.train.RMSPropOptimizer(self.config.learning_rate, decay=self.config.decay, momentum=self.config.momentum).minimize(self.loss)

            self.test_loss = self.test()

            # summary data
            tf.scalar_summary('training_loss', self.loss)
            tf.scalar_summary('test_loss', self.test_loss)
            tf.image_summary('train_img', tf.reshape(255*tf.transpose(self.final_outputs, perm=[0,2,1]), [self.config.batch_size, self.lstmgan_input.feature_vector_size, self.config.num_steps, 1]), max_images=10)
            tf.image_summary('gen_img', tf.reshape(255*tf.clip_by_value(tf.transpose(self.final_outputs, perm=[0,2,1]), 0, 1), [self.config.batch_size, self.lstmgan_input.feature_vector_size, self.config.num_steps, 1]), max_images=10)

            self.summary = tf.merge_all_summaries()
            self.train_writer = tf.train.SummaryWriter(self.summary_dir, self.session.graph)

    def train_batch(self, xbatch, ybatch):
        initial_state = np.zeros((self.config.batch_size, 2*self.config.num_layers*self.config.hidden_size))
        loss, _ = self.session.run([self.loss, self.train_op], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})
        return loss

    def train_epoch(self):
        for i in np.random.permutation(self.lstm_input.epoch_size):
            x, y = reader.get_batch(self.lstm_input.X, self.lstm_input.Y, i)
            loss = self.train_batch(x, y)
        return loss

    def train(self, saver=None, model_file=None, save_iter=None, test_iter=None):
        for i in range(self.config.max_epoch):
            loss = self.train_epoch()
            print "Epoch " + str(i) + ": " + str(loss)

            if save_iter != None and i%save_iter == 0:
                if saver != None and model_file != None:
                    saver.save(self.session, model_file + '_' + str(i/save_iter) + '.ckpt')

            if test_iter != None and i%test_iter == 0:
                test_loss, summary = self.session.run([self.test_loss, self.summary], feed_dict={})
                self.train_writer.add_summary(summary)

    def test(self):
        X = self.lstm_input.test_X
        Y = self.lstm_input.test_Y
        initial_state = np.zeros((self.lstm_input.test_size, 2*self.config.num_layers*self.config.hidden_size))
        next_lstm_state = self.session.run([self.lstm_new_state], feed_dict = {self.xbatch: X, self.initial_state: initial_state})

        out = X[:,-1,:]
        gen_Y = np.zeros(Y.shape) # shape = (test_size, num_steps, feature_vector_size)
        for i in range(gen_Y.shape[1]):
            out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.xbatch: out, self.initial_state: next_lstm_state})
            gen_Y[:,i,:] = out

        Y_reshaped = tf.reshape(Y, [-1, self.lstm_input.feature_vector_size])
        gen_Y_reshaped = tf.reshape(gen_Y, [-1, self.lstm_input.feature_vector_size])

        return tf.reduce_mean(tf.nn.l2_loss(tf.sub(gen_Y_reshaped, Y_reshaped)))
        
    def run_step(self, x, init_zero_state=True):
        if init_zero_state:
            init_value = np.zeros((2*self.config.num_layers*self.config.hidden_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.xbatch: [x], self.initial_state: [init_value]})

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

