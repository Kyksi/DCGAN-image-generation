import matplotlib.pyplot as plt

from transform_images import unscale


def view_dataset(images):
    # plot emojis in order
    lines = 12
    f, axarr = plt.subplots(lines, lines, sharex=True, sharey=True, figsize=(12, 12))
    for i in range(lines ** 2):
        a = axarr[i % lines, i // lines]
        img = images[i]
        a.axis("off")
        a.imshow(img)
    plt.subplots_adjust(wspace=0, hspace=0)


def view_samples(epoch, samples, nrows, ncols, figsize=(5, 5)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = unscale(img)
        im = ax.imshow(img, aspect='equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes


def view_epoch_samples(samples, figsize=(5, 5)):
    epochs = len(samples)
    ncols = 12
    nrows = epochs // ncols
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                             sharey=True, sharex=True)
    print(len(samples))
    for ax, s in zip(axes.flatten(), samples):
        ax.axis('off')
        img = s[3]
        img = unscale(img)
        im = ax.imshow(img, aspect='equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes


def view_losses(losses):
    plt.subplots()
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
