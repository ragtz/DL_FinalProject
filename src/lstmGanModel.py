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
    def __init__(self, config, lstmgan_input, session, summary_dir, name="lstm_gan_model"):
        self.scope = name
        self.session = session
        self.summary_dir = summary_dir

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
            self.d1_outputs, self.d1_presig = self.discriminator(self.xbatch)
            self.d_params_num = len(tf.trainable_variables())
            
            # generator
            g_network_output, self.g_outputs, self.lstm_new_state = self.generator(self.xbatch, self.initial_state)
            g_outputs_reshaped = tf.reshape(self.g_outputs[:,self.config.width/2:,:], [-1, self.lstmgan_input.feature_vector_size])

        with tf.variable_scope(self.scope, reuse=True):
            # discriminator generated samples
            self.d2_outputs, self.d2_presig = self.discriminator(tf.clip_by_value(self.g_outputs, 0, 1))

        self.loss = tf.reduce_mean(tf.log(self.d1_outputs) + tf.log(1 - self.d2_outputs) + tf.nn.l2_loss(tf.sub(g_outputs_reshaped, ybatch_reshaped)))

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

        opt = tf.train.RMSPropOptimizer(self.config.d_learning_rate, decay=self.config.d_decay, momentum=self.config.d_momentum)
        grads_and_vars = opt.compute_gradients(self.d_loss, var_list=d_params)

        for g, v in grads_and_vars:
            tf.scalar_summary(v.name, tf.nn.l2_loss(v))
            tf.scalar_summary(v.name + '_grad', tf.nn.l2_loss(g))

        tf.scalar_summary('d_loss', self.d_loss)
        tf.scalar_summary('g_loss', self.g_loss)
        tf.scalar_summary('rec_loss', self.rec_loss)

        tf.histogram_summary('d1_outputs', self.d1_outputs)
        tf.histogram_summary('d2_outputs', self.d2_outputs)

        tf.image_summary('gen_img', tf.reshape(255*tf.clip_by_value(tf.transpose(self.g_outputs, perm=[0,2,1]), 0, 1), [self.config.batch_size, self.lstmgan_input.feature_vector_size, self.config.width, 1]), max_images=5)

        self.train_summary = tf.merge_all_summaries()
        self.test_summary = tf.scalar_summary('test_loss', self.test_loss)
        self.train_writer = tf.train.SummaryWriter(self.summary_dir, self.session.graph)
        
        self.var_names = self.get_var_names(grads_and_vars)
        self.var_norms = self.get_var_norms(grads_and_vars)
        self.grad_norms = self.get_grad_norms(grads_and_vars)
        
        self.grad_norm = tf.global_norm([gv[0] for gv in grads_and_vars])
        self.train_d = opt.apply_gradients(grads_and_vars)

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

    def discriminator(self, xbatch):
        #xbatch_reshaped = tf.reshape(xbatch, [-1, self.config.width, self.lstmgan_input.feature_vector_size, 1])
        xbatch_reshaped = tf.reshape(xbatch, [-1, self.config.width*self.lstmgan_input.feature_vector_size])

        fc1_shape = [self.config.width*self.lstmgan_input.feature_vector_size, 262144]
        fc1_W = tf.get_variable("fc1_W", fc1_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc1_B = tf.get_variable("fc1_B", (fc1_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc1 = tf.nn.relu(tf.matmul(xbatch_reshaped, fc1_W) + fc1_B)

        fc2_shape = [262144, 262144]
        fc2_W = tf.get_variable("fc2_W", fc2_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc2_B = tf.get_variable("fc2_B", (fc2_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_B)

        fc3_shape = [262144, 262144]
        fc3_W = tf.get_variable("fc3_W", fc3_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc3_B = tf.get_variable("fc3_B", (fc3_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc3 = tf.nn.relu(tf.matmul(fc2, fc3_W) + fc3_B)

        fc4_shape = [262144, 32768]
        fc4_W = tf.get_variable("fc4_W", fc4_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc4_B = tf.get_variable("fc4_B", (fc4_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc4 = tf.nn.relu(tf.matmul(fc3, fc4_W) + fc4_B)

        fc5_shape = [32768, 4096]
        fc5_W = tf.get_variable("fc5_W", fc5_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc5_B = tf.get_variable("fc5_B", (fc5_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc5 = tf.nn.relu(tf.matmul(fc4, fc5_W) + fc5_B)

        fc6_shape = [4096, 1]
        fc6_W = tf.get_variable("fc6_W", fc6_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc6_B = tf.get_variable("fc6_B", (fc6_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc6 = tf.sigmoid(tf.matmul(fc5, fc6_W) + fc6_B)

        return fc6, 0

        '''
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
        '''

        '''
        fc2_shape = [w*h*128, 1]
        fc2_W = tf.get_variable("fc2_W", fc2_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc2_B = tf.get_variable("fc2_B", (fc2_shape[-1],), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        fc2 = tf.sigmoid(tf.matmul(tf.reshape(conv3_P, [-1, fc2_shape[0]]), fc2_W) + fc2_B)
        '''
        return fc2, fc2_presig

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

    def train_batch(self, xbatch, ybatch, losses, var_names, var_norms, grad_norms, d1_labels, d2_labels):
        initial_state = np.zeros((self.config.batch_size, 2*self.config.num_layers*self.config.lstm_size))
        
        # train discriminator
        d_loss, _, d1_outputs, d2_outputs, summary, v_ns, v_nms, g_nms = self.session.run([self.d_loss, self.train_d, self.d1_outputs, self.d2_outputs, self.train_summary, self.var_names, self.var_norms, self.grad_norms], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})

        self.train_writer.add_summary(summary)

        # train generator
        g_loss, _ = self.session.run([self.g_loss, self.train_g], feed_dict={self.xbatch: xbatch, self.ybatch: ybatch, self.initial_state: initial_state})
        #g_loss = 0

        #print d1_outputs[:8], d2_outputs[:8]
        
        losses.append([d_loss, g_loss])
        var_names.append(v_ns)
        var_norms.append(v_nms)
        grad_norms.append(g_nms)
        d1_labels.append(d1_outputs)
        d2_labels.append(d2_outputs)
        
        return d_loss, g_loss, d1_outputs, d2_outputs, losses, var_names, var_norms, grad_norms, d1_labels, d2_labels

    def train_epoch(self, losses, var_names, var_norms, grad_norms, d1_labels, d2_labels):
        for i in np.random.permutation(self.lstmgan_input.epoch_size):
            x, y = reader.get_batch(self.lstmgan_input.X, self.lstmgan_input.Y, i)
            d_loss, g_loss, d1_outputs, d2_outputs, losses, var_names, var_norms, grad_norms, d1_labels, d2_labels = self.train_batch(x, y, losses, var_names, var_norms, grad_norms, d1_labels, d2_labels)
        return d_loss, g_loss, d1_outputs, d2_outputs, losses, var_names, var_norms, grad_norms, d1_labels, d2_labels

    def train(self, test_iter=None):
        losses = []
        var_names = []
        var_norms = []
        grad_norms = []
        d1_labels = []
        d2_labels = []
        for i in range(self.config.max_epoch):
            d_loss, g_loss, d1_outputs, d2_outputs, losses, var_names, var_norms, grad_norms, d1_labels, d2_labels = self.train_epoch(losses, var_names, var_norms, grad_norms, d1_labels, d2_labels)
            #losses.append([i, d_loss, g_loss])
            print "Epoch " + str(i) + ": " + str(d_loss) + ", " + str(g_loss)
            #print d1_outputs[:5].T
            #print d2_outputs[:5].T
            
            #np.savetxt('test_losses_adv_rec.csv', np.array(losses), delimiter=',')
            #np.savetxt('test_var_names.csv', np.array(var_names), delimiter=',')
            #np.savetxt('test_var_norms_adv_rec.csv', np.array(var_norms), delimiter=',')
            #np.savetxt('test_grad_norms_adv_rec.csv', np.array(grad_norms), delimiter=',')
            #np.savetxt('test_d1_labels_adv_rec.csv', np.array(d1_labels), delimiter=',')
            #np.savetxt('test_d2_labels_adv_rec.csv', np.array(d2_labels), delimiter=',')

            if test_iter != None and i%test_iter == 0:
                X = self.lstmgan_input.test_X
                Y = self.lstmgan_input.test_Y
                initial_state = np.zeros((self.lstmgan_input.test_size, 2*self.config.num_layers*self.config.lstm_size))
                loss = self.session.run(self.test_loss, feed_dict = {self.xbatch: X, self.ybatch: Y, self.initial_state: initial_state})
                summary = self.session.run(self.test_summary, feed_dict={self.test_loss: loss})
                self.train_writer.add_summary(summary)
            
        #np.savetxt('test_losses.csv', np.array(losses), delimiter=',')

    def run_step(self):
        if init_zero_state:
            init_value = np.zeros((2*self.config.num_layers*self.config.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.g_outputs, self.lstm_new_state], feed_dict={self.xbatch: [x], self.initial_state: [init_value]})

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

