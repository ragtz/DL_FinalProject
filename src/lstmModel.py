from DL_FinalProject.src import reader
import tensorflow as tf
import numpy as np

class LSTMInput(object):
    def __init__(self, config, data_path, name=None):
        self.data = np.load(data_path)
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.epoch_size = reader.epoch_size(self.data, self.batch_size, self.num_steps)
        self.input, self.targets = reader.img_producer(self.data, self.batch_size, self.num_steps, shuffle=True)

class LSTMModel(object):
    def __init__(self, config, lstm_input, session, name="lstm_model"):
        self.scope = name
        self.session = session

        self.config = config
        self.lstm_input = lstm_input

        with tf.variable_scope(self.scope):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0, state_is_tuple=True)
            self.lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers, state_is_tuple=True)

            output, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.lstm_input.input, dtype=tf.float32)

            self.loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(output, self.lstm_input.targets)))
            self.train_op = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

    def train_epoch(self):
        for i in range(self.lstm_input.epoch_size):
            self.session.run([self.train_op])
            print "Loss:", self.loss

    def run_step(self):
        pass

