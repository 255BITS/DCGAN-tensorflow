
        output = c2d_reshape3 
        #c2d_reshape = tf.reshape(c2d, [self.batch_size, WAV_HEIGHT,WAV_WIDTH, DIMENSIONS])
        #output = self.g_bn1(c2d_reshape3)
        print(c2d_reshape3)
        c2d_reshape3_reshaped = tf.reshape(c2d_reshape3, [self.batch_size, WAV_WIDTH*WAV_HEIGHT*DIMENSIONS])
        #output = fully_connected(c2d_reshape3_reshaped, WAV_WIDTH*WAV_HEIGHT*DIMENSIONS, scope='g_fc')
        print("OUTPUT IS", output.get_shape())
        #output = c2d_reshape3
        output = tf.reshape(output, [self.batch_size, WAV_WIDTH, WAV_HEIGHT, DIMENSIONS])
        return tensorflow_wav.scale_up(output)
