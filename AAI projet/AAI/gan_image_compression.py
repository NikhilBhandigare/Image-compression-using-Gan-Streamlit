# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2

# Define the generator for image compression
def build_image_generator(latent_dim, input_shape):
    input_noise = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(input_noise)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(np.prod(input_shape), activation='sigmoid')(x)
    generated_image = Reshape(input_shape)(x)
    generator = Model(input_noise, generated_image)
    return generator

# Define the discriminator for image compression
def build_image_discriminator(input_shape):
    input_image = Input(shape=input_shape)
    x = Flatten()(input_image)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    validity = Dense(1, activation='sigmoid')(x)
    discriminator = Model(input_image, validity)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return discriminator

# Define GAN model for image compression
def build_image_gan(image_generator, image_discriminator):
    image_discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    compressed_image = image_generator(gan_input)
    validity = image_discriminator(compressed_image)
    gan = Model(gan_input, validity)
    gan.compile(loss='binary_crossentropy', optimizer=Adam())
    return gan

# Function to train GAN for image compression
def train_image_gan(image_generator, image_discriminator, image_gan, images, epochs, batch_size):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    for epoch in range(epochs):
        # Select a random batch of images
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images = images[idx]
        # Generate a batch of compressed images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        compressed_images = image_generator.predict(noise)
        # Train the discriminator
        d_loss_real = image_discriminator.train_on_batch(real_images, valid)
        d_loss_fake = image_discriminator.train_on_batch(compressed_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # Train the generator
        g_loss = image_gan.train_on_batch(noise, valid)
        # Print progress
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")

# Define video compression function
def compress_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_path = 'compressed_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Perform image compression on each frame
        compressed_frame = compress_decompress_image(frame)  # Call image compression function
        out.write(compressed_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

# Example usage
latent_dim = 100
input_shape = (28, 28, 1)  # Example image shape, adjust as needed
image_generator = build_image_generator(latent_dim, input_shape)
image_discriminator = build_image_discriminator(input_shape)
image_gan = build_image_gan(image_generator, image_discriminator)
# Train the GAN with your image data
# train_image_gan(image_generator, image_discriminator, image_gan, images, epochs, batch_size)

# Example video compression usage
video_path = 'input_video.mp4'
compressed_video_path = compress_video(video_path)
print("Compressed video saved at:", compressed_video_path)
