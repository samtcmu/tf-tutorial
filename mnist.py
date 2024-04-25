import numpy as np
import sys
import tensorflow as tf


def PrintMnistExample(pixels):
    """Pretty prints `pixels` to the terminal."""
    print(" -" + ("-" * len(pixels[0]) * 3) + " ")
    for r in range(len(pixels)):
        print("| ", end="")
        for c in range(len(pixels[r])):
            pixel = "  "
            if pixels[r][c] != 0:
                pixel = f"{pixels[r][c]:02x}"
            print(f"{pixel:s} ", end="")
        print("|")
    print(" -" + ("-" * len(pixels[0]) * 3) + " ")


def ReadPixels():
    """Reads handcrafted MNIST example from stdin and returns them as a
    28x28 tensor."""
    pixels = np.zeros([28, 28], dtype=np.int64)
    r, c = 0, 0
    for line in sys.stdin:
        if line[0] != "|":
            continue
        row = line.split("|")[1][1:]
        for c in range(28):
            pixel = row[3*c:(3*c) + 2]
            if pixel == "  ":
                pixel = "00"
            pixels[r][c] = int(pixel, 16)
        r += 1
    return pixels


def main():
    # Prevent numpy from aggressively linewrapping printed tensors.
    np.set_printoptions(linewidth=300)

    # Fetch the MNIST hand written digits dataset.
    mnist = tf.keras.datasets.mnist.load_data()
    train_images, train_labels = mnist[0]
    test_images, test_labels = mnist[1]

    # Print the first n training and test examples. 
    n = 0
    for pixels, label in zip(train_images[0:n], train_labels[0:n]):
        print(pixels)
        print(label)
    for pixels, label in zip(test_images[0:n], test_labels[0:n]):
        print(pixels)
        print(label)

    training_data = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).batch(32)
    test_data = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)).batch(32)

    # Define the neural network model architecture using the Tensorflow
    # Sequential Model API.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((28, 28)),
        tf.keras.layers.Lambda(lambda x: x / 255.0),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Normalization(axis=None, mean=0.0, variance=1.0),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # Define the training procedure for the model.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Train the model.
    model.fit(training_data, epochs=7, validation_data=test_data, verbose=2)

    # Evaluate the performance of the model.
    model.evaluate(test_data, verbose=2)

    # Run the model in inference mode on a particular example.
    pixels = ReadPixels()
    PrintMnistExample(pixels)
    pixels = tf.reshape(pixels, [1, 28, 28])
    raw_inference = model.predict(pixels)[0]
    # print(raw_inference)
    softmaxed_inference = tf.nn.softmax(raw_inference)
    # print(softmaxed_inference.numpy())
    inference = tf.math.argmax(softmaxed_inference)
    print(inference.numpy())


if __name__ == "__main__":
    main()
