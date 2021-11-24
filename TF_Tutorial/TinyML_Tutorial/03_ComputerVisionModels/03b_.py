import tensorflow_datasets as tfds

mnist_data = tfds.load("fashion_mnist")
for item in mnist_data:
     print(item)

