import warnings
warnings.filterwarnings("ignore")
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from skimage import filters, restoration
from skimage import measure
from skimage.morphology import disk

example_file = "images/Dog_face.png"
image = imread(example_file, as_grey = True)
median_filter = filters.rank.median(image, disk(1))
tv_filter = restoration.denoise_tv_chambolle(image, weight = 0.1)
gaussian_filter = filters.gaussian(image, sigma = 0.7)

fig = plt.figure()

for k, (t, F) in enumerate((('Median filter', median_filter),
                            ('TV filter', tv_filter),
                            ('Gaussian filter', gaussian_filter))):
    f = fig.add_subplot(1, 3, k+1)
    plt.axis('off')
    f.set_title(t)
    plt.imshow(F, cmap = cm.gray)

plt.show()
#file_location = images/results/Dog_face_filters.png

#Cropping the image
image_2 = image[5:70, 0:70]
plt.imshow(image_2, cmap=cm.gray)
plt.show()
#file_location = images/results/Dog_face_crop.png
#Cropping is one way to ensure that the images are the correct size for analysis

#Another way
image_3 = resize(image_2, (30, 30), mode = 'edge')
plt.imshow(image_3, cmap=cm.gray)
print("data type: %s, shape: %s" % (type(image_3), image_3.shape))
plt.show()

'''
Currently image_3 is an array of 30 pixels by 30 pixels,
so we can't store it in a dataset because dataset row is always single dimension
now we will flatten the image to convert it into an array of 900 elements
'''

image_row = image_3.flatten()
print("data type: %s, shape: %s" % (type(image_row), image_row.shape))

#Extracting visual features
contours = measure.find_contours(image, 0.55)
plt.imshow(image, cmap=cm.gray)
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth = 2)
    plt.axis('image')
    plt.show()
