import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq
def discriminator(input):
    with tf.variable_scope("lstm_discriminator"):
        cell_input = []
        zeros = []
        vocab_size = int(input.get_shape()[1])
        print('input shape is', input.get_shape())
        for i in range(vocab_size):
            slice = tf.slice(input, [0, i], [-1, 1])
            cell_input.append(slice)
            zeros.append(tf.zeros_like(slice))
        memory = 32
        cell = rnn_cell.BasicLSTMCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*2)
        output, state = seq2seq.basic_rnn_seq2seq(cell_input, zeros, stacked_cell)
        softmax_w = tf.get_variable("softmax_w", [memory*vocab_size, vocab_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
        softmax_b = tf.get_variable("softmax_b", [vocab_size], initializer=tf.constant_initializer(0))
        outputs = tf.concat(1, output)
        logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
        #print("Output shape is", output, state)
        return tf.reduce_sum(tf.nn.sigmoid(logits))
