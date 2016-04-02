import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq
from ops import linear
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
        memory = 4
        labels = [tf.constant(x/memory, dtype=tf.float32) for x in range(memory)]
        print('labels is', labels)
        cell = rnn_cell.BasicLSTMCell(memory)
        stacked_cell = rnn_cell.MultiRNNCell([cell]*1)
        logits, state = seq2seq.basic_rnn_seq2seq(cell_input, zeros, stacked_cell)

        weights = [tf.ones_like(labels_t, dtype=tf.float32)
                           for labels_t in labels]


        #loss = seq2seq.sequence_loss(logits, labels, weights)
        logits_ = tf.concat(1, logits)
        loss = tf.reduce_sum(tf.square(tf.nn.softmax(logits_)))
        #print("Output shape is", output, state)
        return loss
