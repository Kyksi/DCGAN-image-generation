from load_dataset import load_dataset
from train import *

# hyper params
real_size = (64, 64, 3)
z_size = 64
learning_rate_d = 0.00006
learning_rate_g = 0.0001
batch_size = 512
epochs = 800
beta1 = 0.6

if __name__ == '__main__':
    images = load_dataset()
    logger.logging.info('Training dataset: collection of {len} images'.format(len=len(images)))
    view_dataset(images)
    plt.savefig('plots/dataset.png')

    logger.logging.info('Train process with hyper params: \n'
                        'real_size: {real_size} \n'
                        'z_size: {z_size} \n'
                        'learning_rate_d: {learning_rate_d} \n'
                        'learning_rate_g: {learning_rate_g} \n'
                        'batch_size: {batch_size} \n'
                        'epochs: {epochs} \n'
                        'beta1: {beta1} \n'
                        .format(real_size=real_size, z_size=z_size, learning_rate_d=learning_rate_d,
                                learning_rate_g=learning_rate_g, batch_size=batch_size, epochs=epochs, beta1=beta1)
                        )

    # Create the network
    net = GAN(real_size, z_size, learning_rate_d, learning_rate_g, beta1=beta1)

    # Load the data and train the network here
    losses, samples = train(net, images, epochs, batch_size, z_size, print_every=10, show_every=25, figsize=(10, 5))

    _ = view_epoch_samples(samples)
    plt.savefig('plots/epoch-samples.png')

    _ = view_losses(np.array(losses))
    plt.savefig('plots/training-losses.png')
