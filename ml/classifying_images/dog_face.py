#importing the required libraries
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#%matplotlib inline

example_file = ("images/Dog_face.png") #store the image as string in the example_file
image = imread(example_file, as_grey = True) #imread() method reads the string and as_grey tells the method to convert the image to gray scale
plt.imshow(image, cmap = cm.gray) #renders the image and uses a gray scale color map
plt.show() #this displays the image to us
#file_location = images/results/Dog_face_1.png

