import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
from glob import glob
import random
from pathlib import Path

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class LicensePlateVAE(Model):
    def __init__(self, img_height=64, img_width=200, latent_dim=128, **kwargs):
        super(LicensePlateVAE, self).__init__(**kwargs)
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self._build_model()

    def _build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.img_height, self.img_width, 1), name='encoder_input')

        x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
        x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)

        shape_before_flatten = x.shape[1:]
        x = layers.Flatten()(x)

        # Latent space
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

        # Encoder model
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        h_dim, w_dim, c_dim = shape_before_flatten

        x = layers.Dense(h_dim * w_dim * c_dim, activation='relu')(latent_inputs)
        x = layers.Reshape((h_dim, w_dim, c_dim))(x)

        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)

        decoder_outputs = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x, y = data  # noisy images, clean images
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(y, reconstruction),
                    axis=[1, 2]
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

class LicensePlateAugmenter:
    def __init__(self, img_height=64, img_width=200):
        self.img_height = img_height
        self.img_width = img_width

    def add_gaussian_noise(self, image, mean=0, sigma=0.1):
        if len(image.shape) == 2:
            row, col = image.shape
            gauss = np.random.normal(mean, sigma, (row, col))
            noisy = image + gauss
        else:
            row, col, ch = image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss
        return np.clip(noisy, 0, 1)

    def add_salt_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        noisy = np.copy(image)
        salt_mask = np.random.random(image.shape) < salt_prob
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy[salt_mask] = 1.0
        noisy[pepper_mask] = 0.0
        return noisy

    def add_motion_blur(self, image, size=7, angle=45):
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = cv2.warpAffine(kernel,
                               cv2.getRotationMatrix2D((size/2, size/2), angle, 1.0),
                               (size, size))
        kernel = kernel / size
        if len(image.shape) == 3 and image.shape[2] == 1:
            blurred = cv2.filter2D(image.squeeze(), -1, kernel).reshape(image.shape)
        else:
            blurred = cv2.filter2D(image, -1, kernel)
        return np.clip(blurred, 0, 1)

    def apply_random_noise(self, image):
        noise_types = [
            (self.add_gaussian_noise, {}),
            (self.add_salt_pepper_noise, {}),
            (self.add_motion_blur, {})
        ]
        noise_func, params = random.choice(noise_types)
        return noise_func(image, **params)

def load_and_preprocess_images(data_dir, img_height=64, img_width=200):
    """Load and preprocess images from directory"""
    image_paths = sorted(glob(os.path.join(data_dir, "*.png")))
    images = []
    labels = []

    for img_path in image_paths:
        label = os.path.splitext(os.path.basename(img_path))[0]
        labels.append(label)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype('float32') / 255.0
        images.append(img)

    return np.array(images).reshape(-1, img_height, img_width, 1), labels

def create_noisy_dataset(images, output_dir, augmenter):
    """Create and save noisy versions of images"""
    os.makedirs(output_dir, exist_ok=True)
    noisy_images = []

    for i, img in enumerate(images):
        noisy_img = augmenter.apply_random_noise(img)
        noisy_images.append(noisy_img)
        
        # Save noisy image
        noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"noisy_{i}.png"), noisy_img_uint8)

    return np.array(noisy_images)

def visualize_results(original_images, noisy_images, denoised_images, n=5, save_path=None):
    """Visualize original, noisy, and denoised images"""
    plt.figure(figsize=(15, 5))

    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(original_images[i].reshape(original_images.shape[1], original_images.shape[2]), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, n, i + n + 1)
        plt.imshow(noisy_images[i].reshape(noisy_images.shape[1], noisy_images.shape[2]), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        plt.subplot(3, n, i + 2*n + 1)
        plt.imshow(denoised_images[i].reshape(denoised_images.shape[1], denoised_images.shape[2]), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Configuration
    IMG_HEIGHT = 64
    IMG_WIDTH = 200
    LATENT_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Paths
    original_data_dir = "path/to/original/images"  # Update this path
    noisy_data_dir = "path/to/save/noisy/images"   # Update this path
    model_save_dir = "path/to/save/model"          # Update this path
    
    # Create directories
    os.makedirs(noisy_data_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    images, labels = load_and_preprocess_images(original_data_dir, IMG_HEIGHT, IMG_WIDTH)
    
    # Split into train and test sets
    train_size = int(0.8 * len(images))
    train_images = images[:train_size]
    test_images = images[train_size:]
    
    # Create noisy dataset
    print("Creating noisy dataset...")
    augmenter = LicensePlateAugmenter(IMG_HEIGHT, IMG_WIDTH)
    noisy_train_images = create_noisy_dataset(train_images, noisy_data_dir, augmenter)
    
    # Create and compile VAE model
    print("Creating and compiling VAE model...")
    vae = LicensePlateVAE(IMG_HEIGHT, IMG_WIDTH, LATENT_DIM)
    vae.compile(optimizer=Adam(learning_rate=0.0001))
    
    # Train the model
    print("Training VAE model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = vae.fit(
        noisy_train_images,
        train_images,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Save the model
    print("Saving model...")
    vae.save(os.path.join(model_save_dir, "vae_model"))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    noisy_test_images = create_noisy_dataset(test_images, os.path.join(noisy_data_dir, "test"), augmenter)
    denoised_test_images = vae.predict(noisy_test_images)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(
        test_images[:5],
        noisy_test_images[:5],
        denoised_test_images[:5],
        save_path=os.path.join(model_save_dir, "denoising_results.png")
    )

if __name__ == "__main__":
    main() 