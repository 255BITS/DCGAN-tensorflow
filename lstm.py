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
        memory = 16
        cell = rnn_cell.BasicLSTMCell(memory)
        output, state = seq2seq.basic_rnn_seq2seq(cell_input, zeros, cell)
        softmax_w = tf.get_variable("softmax_w", [memory*vocab_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        outputs = tf.concat(1, output)
        logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
        print("Output shape is", output, state)
        return tf.nn.sigmoid(logits)#tf.nn.softmax(logits)
