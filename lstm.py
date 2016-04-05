import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq
from ops import linear
def discriminator(input):
    with tf.variable_scope("lstm_discriminator"):
        cell_input = [input]
        zeros = [tf.zeros_like(input)]
        vocab_size = int(input.get_shape()[1])
        memory = 256
        cell = rnn_cell.BasicLSTMCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*2)
        logits, state = seq2seq.basic_rnn_seq2seq(cell_input, zeros, stacked_cell)

        #labels = [tf.constant(x/memory, dtype=tf.float32) for x in range(memory)]
        #weights = [tf.ones_like(labels_t, dtype=tf.float32)
        #                   for labels_t in labels]


        #loss = seq2seq.sequence_loss(logits, labels, weights)
        logits_ = tf.concat(1, logits)
        print("logits_", logits_)
        # if it's a repeat, one of the memory cells should fire harsh
        w = tf.get_variable('d_softmax_w', [memory, vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        b = tf.get_variable('d_softmax_b', [vocab_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
        wx_b = tf.nn.xw_plus_b(logits_, w, b)
        print('wx_b', wx_b)
        return 6-tf.nn.relu(wx_b)
        #is_repeat = tf.reduce_max(tf.square(tf.nn.softmax(wx_b)), 1)
        #print("Output shape is", output, state)

        # block repeats
        #return (1-is_repeat)

def generator(input):
    with tf.variable_scope("lstm_generator"):
        cell_input = [input]
        zeros = [tf.zeros_like(input)]
        vocab_size = int(input.get_shape()[1])
        memory = 256
        print('input',input)
        cell = rnn_cell.BasicLSTMCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*2)
        logits, state = seq2seq.basic_rnn_seq2seq(cell_input, zeros, stacked_cell)

        #labels = [tf.constant(x/memory, dtype=tf.float32) for x in range(memory)]
        #weights = [tf.ones_like(labels_t, dtype=tf.float32)
        #                   for labels_t in labels]


        #loss = seq2seq.sequence_loss(logits, labels, weights)
        logits_ = tf.concat(1, logits)
        print("logits_", logits_)
        # if it's a repeat, one of the memory cells should fire harsh
        w = tf.get_variable('g_softmax_w', [memory, vocab_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        b = tf.get_variable('g_softmax_b', [vocab_size], dtype=tf.float32, initializer=tf.constant_initializer(0))
        wx_b = tf.nn.xw_plus_b(logits_, w, b)
        softmax = tf.square(tf.nn.softmax(wx_b))
        print('wx_b', wx_b, softmax)
        return softmax
        #is_repeat = tf.reduce_max(tf.square(tf.nn.softmax(wx_b)), 1)
        #print("Output shape is", output, state)

