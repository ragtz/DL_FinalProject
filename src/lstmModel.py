from DL_FinalProject.src import reader
import tensorflow as tf
import numpy as np

class LSTMInput(object):
    def __init__(self, config, data_path, name=None):
        self.data = np.load(data_path)
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.feature_vector_size = reader.feature_vector_size(self.data)
        self.epoch_size = reader.epoch_size(self.data, self.batch_size, self.num_steps)
        self.X, self.Y = reader.batch_data_array(self.data, self.batch_size, self.num_steps)

class LSTMModel(object):
    def __init__(self, config, lstm_input, session, name="lstm_model"):
        self.scope = name
        self.session = session

        self.config = config
        self.lstm_input = lstm_input

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

            self.ybatch = tf.placeholder(tf.float32, (None, None, self.lstm_input.feature_vector_size))
            ybatch_reshaped = tf.reshape(self.ybatch, [-1, self.lstm_input.feature_vector_size])

            self.loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(network_output, ybatch_reshaped)))
            self.train_op = tf.train.RMSPropOptimizer(self.config.learning_rate, decay=self.config.decay, momentum=self.config.momentum).minimize(self.loss)

    def train_batch(self, xbatch, ybatch):
        initial_state = np.zeros((self.config.batch_size, 2*self.config.num_layers*self.config.hidden_size))
        loss, _ = self.session.run([self.loss, self.train_op], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})
        return loss

    def train_epoch(self):
        for i in np.random.permutation(self.lstm_input.epoch_size):
            x, y = reader.get_batch(self.lstm_input.X, self.lstm_input.Y, i)
            loss = self.train_batch(x, y)
        return loss

    def train(self):
        for i in range(self.config.max_epoch):
            loss = self.train_epoch()
            print "Epoch " + str(i) + ": " + str(loss)

    def run_step(self):
        pass

