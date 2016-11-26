
class LSTMGANFDConfig(object):
    d_learning_rate = 0.00001
    d_decay = 0.9
    d_momentum = 0
    d_w = 1.0

    g_learning_rate = 0.0001
    g_decay = 0.9
    g_momentum = 0
    g_w = 1.0

    lstm_size = 512
    num_layers = 3

    fc_size = 1024

    batch_size = 64
    width = 200

    test_width = 200

    max_epoch = 250
