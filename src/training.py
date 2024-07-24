import tensorflow as tf

def train_model(model, train_data, train_labels, epochs=10, batch_size=32):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
    return history

def train_gan(generator, discriminator, gan, data, noise_dim, epochs=10000, batch_size=64, save_interval=1000):
    import numpy as np
    import matplotlib.pyplot as plt

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_images = data[idx]

        noise = np.random.normal(0, 1, (half_batch, noise_dim))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
            save_images(epoch, generator, noise_dim)

def save_images(epoch, generator, noise_dim):
    import numpy as np
    import matplotlib.pyplot as plt

    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, noise_dim))
    gen_images = generator.predict(noise)

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_images[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig(f"images/mnist_{epoch}.png")
    plt.close()