from DL_FinalProject.src import reader
import tensorflow as tf
import numpy as np

class LSTMGANInput(object):
    def __init__(self, config, train_data_path, test_data_path, name=None):
        self.data = np.load(train_data_path)
        self.test_data = np.load(test_data_path)
        self.feature_vector_size = reader.feature_vector_size(self.data)
        self.epoch_size = reader.epoch_size(self.data, config.batch_size, config.width)
        self.X, self.Y = reader.batch_data_array(self.data, config.batch_size, config.width)
        self.test_X, self.test_Y = reader.batch_test_data_array(self.test_data, config.test_width)
        self.test_size = self.test_X.shape[0]

class LSTMGANModel(object):
    def __init__(self, config, lstmgan_input, session, summary_dir, dtype, name="lstm_gan_model"):
        self.scope = name
        self.session = session
        self.summary_dir = summary_dir
        self.dtype = dtype

        self.config = config
        self.lstmgan_input = lstmgan_input

        self.lstm_last_state = np.zeros((2*self.config.num_layers*self.config.lstm_size,))
        self.initial_state = tf.placeholder(tf.float32, shape=(None, 2*self.config.num_layers*self.config.lstm_size), name="initial_state")
        
        self.xbatch = tf.placeholder(tf.float32, shape=(None, None, self.lstmgan_input.feature_vector_size), name="xbatch")
        self.ybatch = tf.placeholder(tf.float32, shape=(None, None, self.lstmgan_input.feature_vector_size), name="ybatch")
        #ybatch_reshaped = tf.reshape(self.ybatch, [-1, self.lstmgan_input.feature_vector_size])
        ybatch_reshaped = tf.reshape(self.ybatch[:,self.config.width/2:,:], [-1, self.lstmgan_input.feature_vector_size])

        with tf.variable_scope(self.scope):
            # discriminator real samples
            self.d1_outputs = self.discriminator(self.xbatch, self.initial_state)
            self.d_params_num = len(tf.trainable_variables())
            
            # generator
            g_network_output, self.g_outputs, self.lstm_new_state = self.generator(self.xbatch, self.initial_state)
            g_outputs_reshaped = tf.reshape(self.g_outputs[:,self.config.width/2:,:], [-1, self.lstmgan_input.feature_vector_size])

        with tf.variable_scope(self.scope, reuse=True):
            # discriminator generated samples
            self.d2_outputs = self.discriminator(tf.clip_by_value(self.g_outputs, 0, 1), self.initial_state)

            print len(tf.trainable_variables())

        self.loss = tf.reduce_mean(self.config.d_w*(tf.log(self.d1_outputs) + tf.log(1 - self.d2_outputs)) + self.config.g_w*tf.nn.l2_loss(tf.sub(g_outputs_reshaped, ybatch_reshaped)))

        self.d_loss = -(tf.reduce_mean(tf.log(self.d1_outputs) + tf.log(1 - self.d2_outputs)))
        self.g_loss = tf.reduce_mean(tf.log(1 - self.d2_outputs) + tf.nn.l2_loss(tf.sub(g_outputs_reshaped, ybatch_reshaped)))

        #self.g_loss = tf.reduce_mean(tf.log(1 - self.d2_outputs))
        #self.g_loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(g_outputs_reshaped, ybatch_reshaped)))
        #self.g_loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(g_network_output, ybatch_reshaped)))
        self.rec_loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(g_network_output, tf.reshape(self.ybatch, [-1, self.lstmgan_input.feature_vector_size]))))
        self.test_loss = tf.reduce_mean(tf.nn.l2_loss(tf.sub(g_network_output, tf.reshape(self.ybatch, [-1, self.lstmgan_input.feature_vector_size]))))

        params = tf.trainable_variables()
        d_params = params[:self.d_params_num]
        g_params = params[self.d_params_num:]

        '''
        opt = tf.train.RMSPropOptimizer(self.config.d_learning_rate, decay=self.config.d_decay, momentum=self.config.d_momentum)
        grads_and_vars = opt.compute_gradients(self.d_loss, var_list=d_params)

        for g, v in grads_and_vars:
            tf.scalar_summary(v.name, tf.nn.l2_loss(v))
            tf.scalar_summary(v.name + '_grad', tf.nn.l2_loss(g))
        '''

        tf.scalar_summary('d_loss', self.d_loss)
        tf.scalar_summary('g_loss', self.g_loss)
        tf.scalar_summary('rec_loss', self.rec_loss)
        tf.scalar_summary('training_loss', self.loss)
        
        tf.histogram_summary('d1_outputs', self.d1_outputs)
        tf.histogram_summary('d2_outputs', self.d2_outputs)

        tf.image_summary('train_img', tf.reshape(255*tf.transpose(self.ybatch, perm=[0,2,1]), [self.config.batch_size, self.lstmgan_input.feature_vector_size, self.config.width, 1]), max_images=10)
        tf.image_summary('gen_img', tf.reshape(255*tf.clip_by_value(tf.transpose(self.g_outputs, perm=[0,2,1]), 0, 1), [self.config.batch_size, self.lstmgan_input.feature_vector_size, self.config.width, 1]), max_images=10)

        self.train_summary = tf.merge_all_summaries()
        self.test_summary = tf.scalar_summary('test_loss', self.test_loss)
        self.train_writer = tf.train.SummaryWriter(self.summary_dir, self.session.graph)

        #self.train_d = tf.train.RMSPropOptimizer(self.config.d_learning_rate, decay=self.config.d_decay, momentum=self.config.d_momentum).minimize(self.d_loss, var_list=d_params)
        #self.train_g = tf.train.RMSPropOptimizer(self.config.g_learning_rate, decay=self.config.g_decay, momentum=self.config.g_momentum).minimize(self.g_loss, var_list=g_params)

        self.train_d = tf.train.RMSPropOptimizer(self.config.d_learning_rate, decay=self.config.d_decay, momentum=self.config.d_momentum).minimize(-self.loss, var_list=d_params)
        self.train_g = tf.train.RMSPropOptimizer(self.config.g_learning_rate, decay=self.config.g_decay, momentum=self.config.g_momentum).minimize(self.loss, var_list=g_params)

    def get_var_names(self, gv):
        return tf.pack([v.name for _, v in gv])

    def get_var_norms(self, gv):
        return tf.pack([tf.nn.l2_loss(v) for _, v in gv])

    def get_grad_norms(self, gv):
        return tf.pack([tf.nn.l2_loss(g) for g, _ in gv])

    def discriminator(self, xbatch, initial_state=None):
        if self.dtype == 0:
            xbatch_reshaped = tf.reshape(xbatch, [-1, self.config.width, self.lstmgan_input.feature_vector_size, 1])

            conv1_shape = [5,5,1,32]
            conv1_W = tf.get_variable("conv1_W", conv1_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv1_B = tf.get_variable("conv1_B", (conv1_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv1 = tf.nn.relu(tf.nn.conv2d(xbatch_reshaped, conv1_W, strides=[1,1,1,1], padding='SAME') + conv1_B)
            conv1_P = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv2_shape = [5,5,32,64]
            conv2_W = tf.get_variable("conv2_W", conv2_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv2_B = tf.get_variable("conv2_B", (conv2_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1_P, conv2_W, strides=[1,1,1,1], padding='SAME') + conv2_B)
            conv2_P = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv3_shape = [5,5,64,128]
            conv3_W = tf.get_variable("conv3_W", conv3_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv3_B = tf.get_variable("conv3_B", (conv3_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2_P, conv3_W, strides=[1,1,1,1], padding='SAME') + conv3_B)
            conv3_P = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv4_shape = [5,5,128,128]
            conv4_W = tf.get_variable("conv4_W", conv4_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv4_B = tf.get_variable("conv4_B", (conv4_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            conv4 = tf.nn.relu(tf.nn.conv2d(conv3_P, conv4_W, strides=[1,1,1,1], padding='SAME') + conv4_B)
            conv4_P = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            w = int(np.ceil(np.ceil(np.ceil(np.ceil(self.config.width/2)/2)/2)/2))
            h = int(np.ceil(np.ceil(np.ceil(np.ceil(self.lstmgan_input.feature_vector_size/2)/2)/2)/2))
            fc1_shape = [w*h*128, self.config.fc_size]
            fc1_W = tf.get_variable("fc1_W", fc1_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            fc1_B = tf.get_variable("fc1_B", (fc1_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            fc1 = tf.nn.relu(tf.matmul(tf.reshape(conv4_P, [-1, fc1_shape[0]]), fc1_W) + fc1_B)

            fc2_shape = [self.config.fc_size, 1]
            fc2_W = tf.get_variable("fc2_W", fc2_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            fc2_B = tf.get_variable("fc2_B", (fc2_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            fc2_presig = tf.matmul(tf.reshape(fc1, [-1, fc2_shape[0]]), fc2_W) + fc2_B
            fc2 = tf.sigmoid(tf.matmul(tf.reshape(fc1, [-1, fc2_shape[0]]), fc2_W) + fc2_B)
 
            return fc2

        else:
            xbatch = tf.reshape(xbatch, [-1, self.config.width, self.lstmgan_input.feature_vector_size])

            with tf.variable_scope("discriminator"):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.lstm_size, forget_bias=1.0, state_is_tuple=False)
                lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers)

                outputs, lstm_new_state = tf.nn.dynamic_rnn(lstm, xbatch, initial_state=initial_state)

                rnnW_shape = (self.config.lstm_size, 1)
                rnnB_shape = (1,)

                rnn_out_W = tf.get_variable("rnn_out_W", rnnW_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
                rnn_out_B = tf.get_variable("rnn_out_B", rnnB_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

                last_idx = tf.shape(outputs)[1] - 1
                last_output = tf.nn.embedding_lookup(tf.transpose(outputs, [1,0,2]), last_idx)

                last_output_reshaped = tf.reshape(last_output, [-1, self.config.lstm_size])
                network_output = tf.sigmoid(tf.matmul(last_output_reshaped, rnn_out_W) + rnn_out_B)

            return network_output

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
        d_loss, _, d1_outputs, d2_outputs, summary = self.session.run([self.d_loss, self.train_d, self.d1_outputs, self.d2_outputs, self.train_summary], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})

        self.train_writer.add_summary(summary)

        # train generator
        g_loss, _ = self.session.run([self.g_loss, self.train_g], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})

        return d_loss, g_loss

    def train_epoch(self):
        for i in np.random.permutation(self.lstmgan_input.epoch_size):
            x, y = reader.get_batch(self.lstmgan_input.X, self.lstmgan_input.Y, i)
            d_loss, g_loss = self.train_batch(x, y)
        return d_loss, g_loss

    def train(self, saver=None, model_file=None, save_iter=None, test_iter=None):
        for i in range(self.config.max_epoch):
            d_loss, g_loss = self.train_epoch()
            print "Epoch " + str(i) + ": " + str(d_loss) + ", " + str(g_loss)

            if save_iter != None and i%save_iter == 0:
                if saver != None and model_file != None:
                    saver.save(self.session, model_file + '_' + str(i/save_iter) + '.ckpt')

            if test_iter != None and i%test_iter == 0:
                X = self.lstmgan_input.test_X
                Y = self.lstmgan_input.test_Y
                initial_state = np.zeros((self.lstmgan_input.test_size, 2*self.config.num_layers*self.config.lstm_size))
                loss = self.session.run(self.test_loss, feed_dict = {self.xbatch: X, self.ybatch: Y, self.initial_state: initial_state})
                summary = self.session.run(self.test_summary, feed_dict={self.test_loss: loss})
                self.train_writer.add_summary(summary)

    def run_step(self):
        if init_zero_state:
            init_value = np.zeros((2*self.config.num_layers*self.config.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.g_outputs, self.lstm_new_state], feed_dict={self.xbatch: [x], self.initial_state: [init_value]})

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

