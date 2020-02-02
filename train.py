import pickle as pkl

import logger
from gan import *
from transform_images import *
from plot_images import *


def train(net, images, epochs, batch_size, z_size, print_every=10, show_every=30, figsize=(5, 5)):
    saver = tf.train.Saver()
    sample_z = np.random.uniform(-1, 1, size=(72, z_size))

    samples, losses = [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for x in get_batches(images, batch_size):
                steps += 1
                x = [scale(i) for i in x]

                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

                # Run optimizers
                _ = sess.run(net.d_opt, feed_dict={net.input_real: x, net.input_z: batch_z})
                _ = sess.run(net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: x})
                _ = sess.run(net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: x})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: x})
                    train_loss_g = net.g_loss.eval({net.input_z: batch_z})

                    logger.logging.info('Epoch {current}/{all} \t'
                                        'Discriminator Loss: {train_loss_d:.4f}... \t'
                                        'Generator Loss: {train_loss_g:.4f}'
                                        .format(current=e+1, all=epochs,
                                                train_loss_d=train_loss_d, train_loss_g=train_loss_g))

                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

                if steps % show_every == 0:
                    gen_samples = sess.run(
                        generator(net.input_z, 3, reuse=True, is_train=False),
                        feed_dict={net.input_z: sample_z})
                    samples.append(gen_samples)
                    _ = view_samples(-1, samples, 6, 12, figsize=figsize)
                    plt.savefig('plots/epoch-%d.png' % (e+1))

        saver.save(sess, './cFinFheckpoints/generator.ckpt')

    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses, samples
