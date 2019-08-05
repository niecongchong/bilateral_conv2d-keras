from bilateral_conv2d import bilateral_conv2d
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

# params
kernel_size = 7
sigma_space = 75
sigma_color = 75

# read img
original_image = cv2.imread('girl.jpg')[:, :, 0]


# process in keras with tf backend
images = np.reshape(original_image, (1, 450, 512, 1))
images = tf.convert_to_tensor(images.astype(np.float32))


def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


gaussian_kernel = gaussian_kernel_2d_opencv(kernel_size=kernel_size, sigma=sigma_space)

gaussian_kernel = np.array(gaussian_kernel).reshape(kernel_size, kernel_size, 1, 1)
gaussian_kernel = tf.convert_to_tensor(gaussian_kernel.astype(np.float32))

bilateral_keras = bilateral_conv2d(images, gaussian_kernel, (1, 1), (450, 512), sigma=sigma_color)

guassian_keras = tf.nn.conv2d(images, gaussian_kernel, (1, 1, 1, 1), 'SAME')

with tf.Session() as sess:
    bilateral_keras = sess.run(bilateral_keras)
    guassian_keras = sess.run(guassian_keras)


# process in opencv
img_guassian = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), sigma_space)

img_bilater = cv2.bilateralFilter(original_image, kernel_size, sigma_color, sigma_space)


plt.figure()
plt.subplot(231)
plt.title('original_image')
plt.imshow(original_image)
plt.subplot(232)
plt.title('guassian_keras')
plt.imshow(guassian_keras[0, :, :, 0])
plt.subplot(233)
plt.title('bilateral_keras')
plt.imshow(bilateral_keras[0, :, :, 0])
plt.subplot(234)
plt.title('original_image')
plt.imshow(original_image)
plt.subplot(235)
plt.title('guassian_opencv')
plt.imshow(img_guassian)
plt.subplot(236)
plt.title('bilateral_opencv')
plt.imshow(img_bilater)
plt.show()

