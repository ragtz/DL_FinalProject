
class LSTMGANTestConfig(object):
    d_learning_rate = 0.001
    d_decay = 0.9
    d_momentum = 0

    g_learning_rate = 0.001
    g_decay = 0.9
    g_momentum = 0

    lstm_size = 512
    num_layers = 3

    fc_size = 1024

    batch_size = 32
    width = 200

    max_epoch = 5
