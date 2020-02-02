import tensorflow as tf


# inputs
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(dtype=tf.float32, shape=(None, *real_dim), name="input_real")
    inputs_z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim), name="input_z")

    return inputs_real, inputs_z


def generator(z, output_dim, reuse=False, alpha=0.2, is_train=True):
    alpha = 0.1
    with tf.variable_scope('generator', reuse=not is_train):
        # First fully connected layer
        fc = tf.layers.dense(inputs=z, units=8 * 8 * 256, activation=None)
        conv1 = tf.reshape(fc, shape=(-1, 8, 8, 256))
        conv1 = tf.layers.batch_normalization(inputs=conv1, training=is_train)
        conv1 = tf.nn.relu(conv1)
        # > 8*8

        conv2 = tf.layers.conv2d_transpose(inputs=conv1, filters=128, kernel_size=5, strides=2, padding='SAME',
                                           activation=None)
        conv2 = tf.layers.batch_normalization(inputs=conv2, training=is_train)
        conv2 = tf.nn.relu(conv2)
        # > 16*16

        conv3 = tf.layers.conv2d_transpose(inputs=conv2, filters=64, kernel_size=5, strides=2, padding='SAME',
                                           activation=None)
        conv3 = tf.layers.batch_normalization(inputs=conv3, training=is_train)
        conv3 = tf.nn.relu(conv3)
        # > 32*32

        # Output layer, 64*64
        logits = tf.layers.conv2d_transpose(inputs=conv3, filters=output_dim, kernel_size=5, strides=2, padding='SAME',
                                            activation=None)

        out = tf.tanh(logits)

    return out


def discriminator(x, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 64*64
        conv1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=5, strides=2, padding='SAME', activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.relu(conv1)
        # > 32*32

        conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=5, strides=2, padding='SAME', activation=None)
        conv2 = tf.layers.batch_normalization(inputs=conv2, training=True)
        conv2 = tf.nn.relu(conv2)
        # > 16*16

        conv3 = tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=5, strides=2, padding='SAME', activation=None)
        conv3 = tf.layers.batch_normalization(inputs=conv3, training=True)
        conv3 = tf.nn.relu(conv3)
        # > 8*8

        flat = tf.reshape(conv3, (-1, 8 * 8 * 256))
        logits = tf.layers.dense(inputs=flat, units=1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits


def model_loss(input_real, input_z, output_dim, alpha=0.2):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param output_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = generator(input_z, output_dim, alpha=alpha)
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)

    smooth_factor = 0.1  # reduce true "correct" by this factor
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss


def model_opt(d_loss, g_loss, learning_rate_d, learning_rate_g, beta1, beta2):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate_d, beta1=beta1, beta2=beta2).minimize(d_loss,
                                                                                                 var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate_g, beta1=beta1, beta2=beta2).minimize(g_loss,
                                                                                                 var_list=g_vars)

    return d_train_opt, g_train_opt


class GAN:
    def __init__(self, real_size, z_size, learning_rate_d, learning_rate_g, alpha=0.2, beta1=0.5, beta2=0.5):
        tf.reset_default_graph()

        self.input_real, self.input_z = model_inputs(real_size, z_size)

        self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z,
                                              real_size[2], alpha=0.2)

        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate_d, learning_rate_g, beta1, beta2)
