import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# Load image
image = cv2.imread('noisy_image.jpg', 0)

# Apply Gaussian Blurring
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Median Blurring
median_blur = cv2.medianBlur(image, 5)

# Apply Bilateral Filtering
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

# Define a simple autoencoder model for denoising
input_img = Input(shape=(image.shape[0], image.shape[1], 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Prepare the image for the autoencoder
image = image.astype('float32') / 255.
image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))

# Train the autoencoder (using the same image as both input and target for simplicity)
autoencoder.fit(image, image, epochs=100, batch_size=1, shuffle=True)

# Apply the autoencoder
denoised_image = autoencoder.predict(image)
denoised_image = np.reshape(denoised_image, (denoised_image.shape[1], denoised_image.shape[2]))

# Display images
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image[0, :, :, 0], cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Gaussian Blurred Image')
plt.imshow(gaussian_blur, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Median Blurred Image')
plt.imshow(median_blur, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Bilateral Filtered Image')
plt.imshow(bilateral_filter, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('Denoised Image with Autoencoder')
plt.imshow(denoised_image, cmap='gray')

plt.show()
