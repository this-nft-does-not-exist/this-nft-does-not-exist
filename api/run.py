import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    # Note: None is the batch size
    assert model.output_shape == (None, 8, 8, 256)

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (20, 20), strides=(4, 4),
              padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (10, 10), strides=(2, 2),
              padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2),
              padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

discriminator = make_discriminator_model()
decision = discriminator(generated_image)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def restore_checkpoint():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


NOISE_DIM = 100


def generate_image(model=generator, seed=None):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    if seed is None:
        seed = tf.random.normal([1, NOISE_DIM])
    predictions = model(seed, training=False)
    im = Image.fromarray((predictions[0].numpy() * 255).astype(np.uint8))
    return im


def generate_and_save_images(model, epoch, test_input):
    im = generate_image(model, test_input)
    im.save('image_at_epoch_{:04d}.png'.format(epoch))
    im.close()


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    num_examples_to_generate = 1

    # You will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

    for epoch in range(epochs):
        start = time.time()

        # Produce images for the GIF as you go
        generate_and_save_images(generator, epoch + 1, seed)
        for image_batch in dataset:
            train_step(image_batch)

        # Save the model every 1 epochs
        if (epoch + 1) % 8 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    return


if __name__ == "__main__":
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        "nft", label_mode=None, image_size=(64, 64), batch_size=32, shuffle=True
    )
    # Normalize the images to [-1, 1]
    train_dataset = train_dataset.map(lambda x: x/255.0)
    train_images_array = []
    for images in train_dataset:
        for i in range(len(images)):
            train_images_array.append(images[i])
    train_images = np.array(train_images_array)
    train_images = train_images.reshape(train_images.shape[0], 64, 64, 3).astype('float32')
    # Batch and shuffle the data
    training_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train(training_dataset, 1024)
