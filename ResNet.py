import tensorflow as tf


class ResNet:
    resnet_size = 0
    net = None
    variable_scope_base = 'resnet'

    def __init__(self, resnet_size):
        self.resnet_size = resnet_size

    # use this function to build ResNet
    def build_resnet(self, input_tensor):
        if self.resnet_size < 20 or (self.resnet_size - 20) % 12 != 0:
            print('incorrect resnet size!')
            return False

        num_conv = (self.resnet_size - 20) // 12 + 1
        print('number of convolution layers per block: %d' % (num_conv * 4))

        with tf.variable_scope(self.variable_scope_base + 'conv1', reuse=tf.AUTO_REUSE):
            self.net = self.build_conv_layer(
                input_tensor=input_tensor,
                filter_depth=32,
                kernel_size=3,
                stride=1,
                has_relu=True)
        print('convolution 1 size: {}'.format(self.net.shape[1:]))

        for i in range(num_conv):
            with tf.variable_scope(self.variable_scope_base + '_conv2_' + str(i + 1), reuse=tf.AUTO_REUSE):
                self.net = self.build_block(self.net, filter_depth=32)
                self.net = self.build_block(self.net, filter_depth=32)
        print('convolution 2 size: {}'.format(self.net.shape[1:]))

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope(self.variable_scope_base + '_conv3_' + str(i + 1), reuse=tf.AUTO_REUSE):
                self.net = self.build_block(self.net, filter_depth=64, down_sample=down_sample)
                self.net = self.build_block(self.net, filter_depth=64)
        print('convolution 3 size: {}'.format(self.net.shape[1:]))

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope(self.variable_scope_base + '_conv4_' + str(i + 1), reuse=tf.AUTO_REUSE):
                self.net = self.build_block(self.net, filter_depth=128, down_sample=down_sample)
                self.net = self.build_block(self.net, filter_depth=128)
        print('convolution 4 size: {}'.format(self.net.shape[1:]))

        with tf.variable_scope(self.variable_scope_base + '_fc', reuse=tf.AUTO_REUSE):
            self.net = tf.reduce_mean(
                input_tensor=self.net,
                axis=[1, 2])
            print('average pooling size: {}'.format(self.net.shape[1:]))

            self.net = tf.layers.flatten(inputs=self.net)
            print('flatten size: {}'.format(self.net.shape[1:]))

            self.net = tf.contrib.layers.fully_connected(inputs=self.net, num_outputs=10)
            print('fc size: {}'.format(self.net.shape[1:]))

            self.net = tf.contrib.layers.softmax(logits=self.net)
            print('softmax size: {}'.format(self.net.shape[1:]))

            print('Build complete.')

        return True

    # use this function to create blocks
    def build_block(self, input_tensor, filter_depth, down_sample=False, projection=False):
        if down_sample:
            input_tensor = tf.layers.max_pooling2d(
                inputs=input_tensor,
                pool_size=2,
                strides=2,
                padding='same')

        conv1 = self.build_conv_layer(
            input_tensor=input_tensor,
            filter_depth=filter_depth,
            kernel_size=3,
            stride=1,
            has_relu=True)

        conv2 = self.build_conv_layer(
            input_tensor=conv1,
            filter_depth=filter_depth,
            kernel_size=3,
            stride=1)

        input_tensor_depth = input_tensor.shape[3].value
        if input_tensor_depth != filter_depth:
            if projection:
                input_layer = self.build_conv_layer(
                    input_tensor=input_tensor,
                    filter_depth=filter_depth,
                    kernel_size=1,
                    stride=2)
            else:
                input_layer = tf.pad(
                    input_tensor,
                    [[0, 0], [0, 0], [0, 0],
                     [0, filter_depth - input_tensor_depth]])
        else:
            input_layer = input_tensor

        output_tensor = conv2 + input_layer
        output_tensor = tf.nn.relu(features=output_tensor)
        #output_tensor = tf.nn.leaky_relu(features=output_tensor, alpha=0.01)

        return output_tensor

    # use this function to create conv layer
    def build_conv_layer(self, input_tensor, filter_depth, kernel_size, stride, has_relu=False):
        conv_layer = tf.layers.conv2d(
            inputs=input_tensor,
            filters=filter_depth,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv_layer = tf.layers.batch_normalization(inputs=conv_layer)

        if has_relu:
            conv_layer = tf.nn.relu(features=conv_layer)
            #conv_layer = tf.nn.leaky_relu(features=conv_layer, alpha=0.01)

        return conv_layer

    # use this function to get resnet
    def get_resnet(self):
        return self.net

    # use this function to set resnet size
    def set_size(self, resnet_size):
        self.resnet_size = resnet_size
