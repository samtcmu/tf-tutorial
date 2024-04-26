import argparse
import datetime
import numpy as np
import random
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


def GetShiftRange(pixels):
    up = 0
    for r in range(len(pixels)):
        if any(pixels[r][c] != 0 for c in range(len(pixels[r]))):
            break
        up += 1

    down = 0
    for r in range(len(pixels) - 1, -1 , -1):
        if any(pixels[r][c] != 0 for c in range(len(pixels[r]))):
            break
        down += 1

    left = 0
    for c in range(len(pixels[0])):
        if any(pixels[r][c] != 0 for r in range(len(pixels))):
            break
        left += 1

    right = 0
    for c in range(len(pixels[0]) - 1, -1, -1):
        if any(pixels[r][c] != 0 for r in range(len(pixels))):
            break
        right += 1

    return (-up, down), (-left, right)


def GenerateShiftedData(images, labels, unit_name='example'):
    if len(images) == 0 or len(labels) == 0 or len(images) != len(labels):
        return None, None

    progress_bar = tf.keras.utils.Progbar(
        len(images), 60, verbose=1, unit_name=unit_name)

    shifted_images = []
    shifted_labels = []
    for pixels, label in zip(images, labels):
        shift_row_range, shift_column_range = GetShiftRange(pixels)
        for dr in range(shift_row_range[0], shift_row_range[1] + 1):
            for dc in range(shift_column_range[0], shift_column_range[1] + 1):
                shifted_pixels = np.roll(pixels, (dr, dc), axis=(0, 1))
                shifted_images.append(shifted_pixels.reshape((1, 28, 28)))
                shifted_labels.append(label.reshape((1, 1)))
        progress_bar.add(1)
    return np.concatenate(shifted_images), np.concatenate(shifted_labels)


def Infer(model, pixels):
    pixels = tf.reshape(pixels, [1, 28, 28])
    raw_inference = model.predict(pixels)[0]
    softmaxed_inference = tf.nn.softmax(raw_inference)
    inference = tf.math.argmax(softmaxed_inference)
    return inference.numpy()


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_model", "-t",
        help="Whether to train a model on the MNIST dataset.",
        default=False,
        action="store_true")
    parser.add_argument(
        "--model_file", "-m",
        help="The path to either the model to load or where the model should "
             "be saved.",
        default="mnist_model.keras",
        type=str)
    flags = parser.parse_args()

    # Prevent numpy from aggressively linewrapping printed tensors.
    np.set_printoptions(linewidth=300)

    # Fetch the MNIST hand written digits dataset.
    mnist = tf.keras.datasets.mnist.load_data()
    training_images, training_labels = mnist[0]
    test_images, test_labels = mnist[1]

    generated_shifted_data = True
    if generated_shifted_data:
        print("Generating shifted training and test data")
        if flags.train_model:
            training_images, training_labels = GenerateShiftedData(
                training_images, training_labels, unit_name="training example")
        test_images, test_labels = GenerateShiftedData(
            test_images, test_labels, unit_name="test example")

    # Print the first n training and test examples. 
    n = 0
    for pixels, label in zip(training_images[0:n], training_labels[0:n]):
        print(pixels)
        print(label)
    for pixels, label in zip(test_images[0:n], test_labels[0:n]):
        print(pixels)
        print(label)

    training_data = tf.data.Dataset.from_tensor_slices(
        (training_images, training_labels)).shuffle(10000).batch(8192)
    test_data = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)).shuffle(10000).batch(8192)

    if flags.train_model:
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
        
        # Set up tensorboard. Run the following command on the command line during
        # training:
        #   python3 -m tensorboard.main --logdir=logs/fit
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)

        # Train the model.
        print(f"Training model: {model.summary()}")
        model.fit(training_data, epochs=7, validation_data=test_data, verbose=1,
                  callbacks=[tensorboard_callback])

        # Evaluate the performance of the model.
        print("Evaluating model:")
        model.evaluate(test_data, verbose=1)

        print(f"Saving model to {flags.model_file:s}")
        model.save(flags.model_file)
    else:
        # This is needed because the model has a Lambda layer.
        tf.keras.config.enable_unsafe_deserialization()

        print(f"Loading model from {flags.model_file:s}")
        model = tf.keras.models.load_model(flags.model_file)
        print(f"Loaded model from {flags.model_file:s}: {model.summary()}")

        print("Evaluating model:")
        model.evaluate(test_data, verbose=1)

    # Run the model in inference mode on a particular example.
    pixels = ReadPixels()
    PrintMnistExample(pixels)
    print(Infer(model, pixels))
    shift_row_range, shift_column_range = GetShiftRange(pixels)
    for i in range(10):
        (dr, dc) = (random.randint(shift_row_range[0], shift_row_range[1]),
                    random.randint(shift_column_range[0], shift_column_range[1]))
        print(f"shift {i:02d}: ({dr:d}, {dc:d})")
        shifted_pixels = np.roll(pixels, (dr, dc), axis=(0, 1))
        PrintMnistExample(shifted_pixels)
        print(Infer(model, shifted_pixels))


if __name__ == "__main__":
    main()
