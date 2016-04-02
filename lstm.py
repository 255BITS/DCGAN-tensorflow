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
        # if it's a repeat, one of the memory cells should fire harsh
        is_repeat = tf.reduce_max(tf.square(tf.nn.softmax(logits_)))
        #print("Output shape is", output, state)

        # block repeats
        return (1-is_repeat)
