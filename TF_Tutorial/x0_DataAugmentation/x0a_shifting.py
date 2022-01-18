from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

WIDTH_SHIFT_RANGE = 0.5
HEIGHT_SHIFT_RANGE = 0.5

# load the image
img = load_img('cat.jpg')

# convert to numpy array
data = img_to_array(img)

# expand dimension to become samples a ONE SAMPLE BATCH
samples = expand_dims(data, 0)

# create image data augmentation generator for 
datagen = ImageDataGenerator(	width_shift_range = WIDTH_SHIFT_RANGE, \
								height_shift_range = HEIGHT_SHIFT_RANGE)

# prepare iterator
it = datagen.flow(samples, batch_size=1)

# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()