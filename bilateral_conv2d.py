import tensorflow as tf
import keras.backend as K


def int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None


def bilateral_conv2d(inputs, kernel, strides, output_shape, sigma=50):
    # data_format = normalize_data_format(data_format)

    stride_row, stride_col = strides
    output_row, output_col = output_shape
    kernel_shape = int_shape(kernel)
    kernel_row, kernel_col, input_filter, filters = kernel_shape

    # result: (b, output_row, output_col, kernel_row * kernel_col * input_filter)
    image_patches = tf.extract_image_patches(inputs,
                                             [1, kernel_row, kernel_col, 1],
                                             [1, stride_row, stride_col, 1],
                                             [1, 1, 1, 1],
                                             padding='SAME')

    # result: (b, output_row, output_col, 1 * 1 * input_filter)
    center_patches = tf.extract_image_patches(inputs,
                                              [1, 1, 1, 1],
                                              [1, stride_row, stride_col, 1],
                                              [1, 1, 1, 1],
                                              padding='VALID')

    image_patches = tf.reshape(image_patches, (-1, kernel_row * kernel_col, input_filter))
    center_patches = tf.reshape(center_patches, (-1, 1, input_filter))

    SqA = tf.square(image_patches)
    sumSqA = tf.reduce_sum(SqA, axis=-1, keepdims=True)

    SqB = tf.square(center_patches)
    sumSqB = tf.reduce_sum(SqB, axis=-1, keepdims=True)
    sumSqBEx = tf.tile(sumSqB, (1, kernel_row * kernel_col, 1))

    ABT = K.batch_dot(image_patches, tf.transpose(center_patches, (0, 2, 1)))

    SqED = sumSqBEx + sumSqA - 2 * ABT

    coefficient_weight = tf.exp(-SqED/(2*sigma**2))  # shape=(b * output_row * output_col, kernel_row * kernel_col, 1)

    coefficient_weight = tf.reshape(coefficient_weight, (-1, output_row, output_col, kernel_row, kernel_col, 1, 1))

    kernel_weights = tf.reshape(kernel, (1, 1, 1, kernel_row, kernel_col, input_filter, filters))

    weights = coefficient_weight * kernel_weights

    image_patches = tf.reshape(image_patches, (-1, kernel_row * kernel_col * input_filter))
    weights = tf.reshape(weights, (-1, kernel_row * kernel_col * input_filter, filters))

    output = K.batch_dot(image_patches, weights)
    output = tf.reshape(output, (-1, output_row, output_col, filters))

    return output
