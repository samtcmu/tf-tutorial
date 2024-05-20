# Python Installation

## Check if Python is installed

Run either of the 2 commands to see if Python is installed:

```shell
python --version
python3 --version
```

You should see something like the following printed to the terminal:

```
Python 3.9.6
```

We are looking for any version of Python 3 to be installed.

## Python language resources

Here are some samples of simple examples of common things in the Python
programmning language. For more, consult your Python book or use Google/Gemini.

```python
# Create a variable and set it to an integer.
x = 314159

# Create a variable and set it to a string.
y = "pi"

# Simple if statement
if x == 2718281828:
  print("mathematical constant: e")
elif x == 314159:
  print("mathematical constant: pi")
else:
  print("unknown constant")

# Boolean operators
if (x == 2718281828) or (x == 314159):
  print("known constant")
if (x == 314159) and (y == "pi"):
  print("both x and y are pi!")
if not ((x % 2) == 0):
  print("not even")
if (x % 2) != 0:
  print("not even")

# Printing a format string (for more info see
# https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals)
# {x:d} prints the variable x as an integer
# {y:s} prints the variable y as a string
print(f"variable x = {x:d} and variable y = {y:s}")

# Create a List and print it
L = ["a", "b", "c", "d", "e"]
print(L)

# Iterate through its elements with a for loop
for l in L:
  print(l)

# Iterate through its elements and keep track of the current index:
for i, l in enumerate(L):
  print(f"index: {i:d} value: {l:s}")

# While loop
i = 0
while True:
  if i > 10:
    break
  if i == 4:
    # Dont' print 4 since it isn't lucky!
    continue
  i += 1

# Simple functions
def Double(a):
  return 2 * a

def SuperSum(a, b):
  return a + b

# Lambda functions
super_sum_lambda = (lambda a, b: a + b)

# Calling functions (and lambda functions)
a = Double(2)
b = SuperSum(a, 2)
c = super_sum_lambda(a, b)
print(f"{a:d}, {b:d}, {c:d}")

```

## Simple python script creation

```python
def main():
  print("hello, world!")

if __name__ == "__main__":
    main()
```

Save this to a file named `hello.py`. You can then run this by running the
follow command in the terminal:

```shell
python3 hello.py
```

## Install Tensorflow

```shell
python3 -m pip install numpy
python3 -m pip install jax
python3 -m pip install tensorflow
python3 -m pip install jaxlib
python3 -m pip install tensorboard
python3 -m pip install --user tensorboard
```

## Machine learning overview
-   [Neural Networks for Machine Learning with Geoffrey Hinton](https://www.youtube.com/playlist?list=PLLssT5z_DsK_gyrQ_biidwvPYCRNGI3iv)
    -   Geoffrey Hinton is a pioneer in machine learning and AI; these lectures
        are the best I have seen
    -   These lectures will give an overview of machine learning.
    -   Watch a few of these and ask me questions if you are confused.

## Tensorflow MNIST tutorial

-   [TensorFlow 2 quickstart for beginners](https://www.tensorflow.org/tutorials/quickstart/beginner)
-   [TensorFlow 2 quickstart for experts](https://www.tensorflow.org/tutorials/quickstart/advanced)
    -   Ignore the name, you should do this tutorial so you know how to deal
        with a tf.data.Dataset object holding your training and test data.
-   [More Tensorflow tutorials if you are curious](https://www.tensorflow.org/tutorials)

## Tensorflow Diamonds dataset

This week I want you to build a model that can predict diamond prices based on
their attributes (carrot weight, clarity, cut, color, etc.). Here is the
dataset to get started:

-   [Tensorflow Diamonds dataset](https://www.tensorflow.org/datasets/catalog/diamonds)
