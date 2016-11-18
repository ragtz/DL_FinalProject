from DL_FinalProject.src import reader
import tensorflow as tf
import numpy as np

class LSTMGANInput(object):
    def __init__(self, config, data_path, name=None):
        self.data = np.load(data_path)
        self.feature_vector_size = reader.feature_vector_size(self.data)
        self.epoch_size = reader.epoch_size(self.data, config.batch_size, config.width)
        self.X, self.Y = reader.batch_data_array(self.data, config.batch_size, config.width)

class LSTMGANModel(object):
    def __init__(self, config, lstmgan_input, session, name="lstm_gan_model"):
        self.scope = name
        self.session = session

        self.config = config
        self.lstmgan_input = lstmgan_input

        self.lstm_last_state = np.zeros((2*self.config.num_layers*self.config.lstm_size,))
        self.initial_state = tf.placeholder(tf.float32, shape=(None, 2*self.config.num_layers*self.config.lstm_size), name="initial_state")
        
        self.xbatch = tf.placeholder(tf.float32, shape=(None, self.config.width, self.lstmgan_input.feature_vector_size), name="xbatch")
        self.ybatch = tf.placeholder(tf.float32, shape=(None, self.config.width, self.lstmgan_input.feature_vector_size), name="ybatch")
        ybatch_reshaped = tf.reshape(self.ybatch, [-1, self.lstmgan_input.feature_vector_size])

        with tf.variable_scope(self.scope):
            # discriminator real samples
            self.d1_outputs = self.discriminator(self.xbatch)
            self.d_params_num = len(tf.trainable_variables())
            
            # generator
            g_network_output, self.g_outputs, self.lstm_new_state = self.generator(self.xbatch, self.initial_state)

        with tf.variable_scope(self.scope, reuse=True):
            # discriminator generated samples
            self.d2_outputs = self.discriminator(self.g_outputs)

        self.d_loss = tf.reduce_mean(tf.log(self.d1_outputs) + tf.log(1 - self.d2_outputs))
        #self.g_loss = tf.reduce_mean(tf.log(1 - self.d2_outputs) + tf.nn.l2_loss(tf.sub(tf.slice(self.d2_outputs), tf.slice(self.ybatch))))
        self.g_loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(g_network_output, ybatch_reshaped)))

        params = tf.trainable_variables()
        d_params = params[:self.d_params_num]
        g_params = params[self.d_params_num:]

        self.train_d = tf.train.RMSPropOptimizer(self.config.d_learning_rate, decay=self.config.d_decay, momentum=self.config.d_momentum).minimize(self.d_loss, var_list=d_params)
        self.train_g = tf.train.RMSPropOptimizer(self.config.g_learning_rate, decay=self.config.g_decay, momentum=self.config.g_momentum).minimize(self.g_loss, var_list=g_params)

    def discriminator(self, xbatch):
        xbatch_reshaped = tf.reshape(xbatch, [-1, self.config.width, self.lstmgan_input.feature_vector_size, 1])

        conv1_shape = [5,5,1,32]
        conv1_W = tf.Variable(tf.random_normal(conv1_shape, stddev=0.01))
        conv1_B = tf.Variable(tf.random_normal((conv1_shape[-1],), stddev=0.01))
        conv1 = tf.nn.relu(tf.nn.conv2d(xbatch_reshaped, conv1_W, strides=[1,1,1,1], padding='SAME') + conv1_B)
        conv1_P = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2_shape = [5,5,32,64]
        conv2_W = tf.Variable(tf.random_normal(conv2_shape, stddev=0.01))
        conv2_B = tf.Variable(tf.random_normal((conv2_shape[-1],), stddev=0.01))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1_P, conv2_W, strides=[1,1,1,1], padding='SAME') + conv2_B)
        conv2_P = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3_shape = [5,5,64,128]
        conv3_W = tf.Variable(tf.random_normal(conv3_shape, stddev=0.01))
        conv3_B = tf.Variable(tf.random_normal((conv3_shape[-1],), stddev=0.01))
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2_P, conv3_W, strides=[1,1,1,1], padding='SAME') + conv3_B)
        conv3_P = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        w = int(np.ceil(np.ceil(np.ceil(self.config.width/2)/2)/2))
        h = int(np.ceil(np.ceil(np.ceil(self.lstmgan_input.feature_vector_size/2)/2)/2))
        fc1_shape = [w*h*128, self.config.fc_size]
        fc1_W = tf.Variable(tf.random_normal(fc1_shape, stddev=0.01))
        fc1_B = tf.Variable(tf.random_normal((fc1_shape[-1],), stddev=0.01))
        fc1 = tf.nn.relu(tf.matmul(tf.reshape(conv3_P, [-1, fc1_shape[0]]), fc1_W) + fc1_B)

        fc2_shape = [self.config.fc_size, 1]
        fc2_W = tf.Variable(tf.random_normal(fc2_shape, stddev=0.01))
        fc2_B = tf.Variable(tf.random_normal((fc2_shape[-1],), stddev=0.01))
        fc2 = tf.nn.relu(tf.matmul(tf.reshape(fc1, [-1, fc2_shape[0]]), fc2_W) + fc2_B)

        return fc2

    def generator(self, xbatch, initial_state):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.lstm_size, forget_bias=1.0, state_is_tuple=False)
        lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers)

        outputs, lstm_new_state = tf.nn.dynamic_rnn(lstm, xbatch, initial_state=initial_state)

        rnn_out_W = tf.Variable(tf.random_normal((self.config.lstm_size, self.lstmgan_input.feature_vector_size), stddev=0.01))
        rnn_out_B = tf.Variable(tf.random_normal((self.lstmgan_input.feature_vector_size,), stddev=0.01))

        outputs_reshaped = tf.reshape(outputs, [-1, self.config.lstm_size])
        network_output = (tf.matmul(outputs_reshaped, rnn_out_W) + rnn_out_B)

        batch_time_shape = tf.shape(outputs)

        final_outputs = tf.reshape(network_output, (batch_time_shape[0], batch_time_shape[1], self.lstmgan_input.feature_vector_size))

        return network_output, final_outputs, lstm_new_state

    def train_batch(self, xbatch, ybatch):
        initial_state = np.zeros((self.config.batch_size, 2*self.config.num_layers*self.config.lstm_size))
        
        # train discriminator
        d_loss, _ = self.session.run([self.d_loss, self.train_d], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})

        # train generator
        #g_loss, _ = self.session.run([self.g_loss, self.train_g], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})
        
        return d_loss, g_loss

    def train_epoch(self):
        for i in np.random.permutation(self.lstmgan_input.epoch_size):
            x, y = reader.get_batch(self.lstmgan_input.X, self.lstmgan_input.Y, i)
            d_loss, g_loss = self.train_batch(x, y)
        return d_loss, g_loss

    def train(self):
        for i in range(self.config.max_epoch):
            d_loss, g_loss = self.train_epoch()
            print "Epoch " + str(i) + ": " + str(d_loss) + ", " + str(g_loss)

    def run_step(self):
        if init_zero_state:
            init_value = np.zeros((2*self.config.num_layers*self.config.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.g_outputs, self.lstm_new_state], feed_dict={self.xbatch: [x], self.initial_state: [init_value]})

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

