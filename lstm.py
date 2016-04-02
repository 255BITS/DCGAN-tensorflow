import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq
from ops import linear
def discriminator(input):
    with tf.variable_scope("lstm_discriminator"):
        cell_input = []
        zeros = []
        vocab_size = int(input.get_shape()[1])
        for i in range(vocab_size):
            slice = tf.slice(input, [0, i], [-1, 1])
            cell_input.append(slice)
            zeros.append(tf.zeros_like(slice))
        memory = 256
        labels = [tf.constant(x/memory, dtype=tf.float32) for x in range(memory)]
        cell = rnn_cell.BasicLSTMCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*2)
        logits, state = seq2seq.basic_rnn_seq2seq(cell_input, zeros, stacked_cell)

        weights = [tf.ones_like(labels_t, dtype=tf.float32)
                           for labels_t in labels]


        #loss = seq2seq.sequence_loss(logits, labels, weights)
        logits_ = tf.concat(1, logits)
        # if it's a repeat, one of the memory cells should fire harsh
        is_repeat = tf.reduce_max(tf.square(tf.nn.softmax(logits_)))
        #print("Output shape is", output, state)

        # block repeats
        return (1-is_repeat)
