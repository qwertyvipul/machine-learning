'''
Eigenfaces is an approach to facial recognition based on the overall appearance
of a face, not on its particular details. It's a less effective technique than
extracting features from the details of an image, yet it works, and you can implement
quickly on your computer.

References:
1. https://en.wikipedia.org/wiki/Eigneface
2. http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html
'''

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
dataset = fetch_olivetti_faces(shuffle=True, random_state=101)
train_faces = dataset.data[:350, :]
test_faces = dataset.data[350:, :]
train_answers = dataset.target[:350]
test_answers = dataset.target[350:]
#print(dataset.DESCR) #prints the description of the dataset

'''
The olivetti dataset consists of 400 photos taken from 40 people (so there are 10
photos of each person). Even though the photos represent the same person, each photo
has been taken at different times during the day, with different light and facial
expressions or details. The images are 64 x 64 pixels, so unfolding all the pixels
into features creates a dataset made of 400 cases and 4,096 variables. Using RandomizedPCA,
you can reduce them to a smaller and more manageable number.
'''

from sklearn.decomposition import RandomizedPCA

n_components = 25
Rpca = RandomizedPCA(n_components=n_components, whiten=True, random_state=101).fit(train_faces)
#print('Explained variance by %i components: %0.3f' %(n_components, np.sum(Rpca.explained_variance_ratio_)))
compressed_train_faces = Rpca.transform(train_faces)
compressed_test_faces = Rpca.transform(test_faces)

'''
The RandomizedPCA class is an approxiamte PCA version, which works better when the dataset
is large (has many rows and variables). The decomposition creates 25(n_components parameter) new variables and
whitening(whiten = TRUE), removing some constant noise (created by textual and photo
granularity) and irrelevant information from images kin a different way from the filters.
The resulting decomposition uses 25 components, which is about 80 percent of the information
held in 4,096 features.
'''

import matplotlib.pyplot as plt
photo = 17 #This is the photo in the test set
#print('We are looking for face id=%i' % test_answers[photo])
#plt.subplot(1, 2, 1)
plt.subplot(2, 2, 1)
plt.axis('off')
plt.title('Unknown face '+str(photo)+' in test set')
plt.imshow(test_faces[photo].reshape(64, 64), cmap=plt.cm.gray, interpolation='nearest')
#plt.show()

'''
After the decomposition of the test set, the example takes the data realtive only to photo
17 and subtracts it from the decomposition of the training set. Now the training set is
made of differences with respect to the example photo. The code squares them (to remove
negative values) and sums them by row, which results in a series of summed errors. The most
similar photos are the ones with the least squared errors, that is, the ones whose differences
are the least
'''

#Just the vector of value components of our photo
mask = compressed_test_faces[photo,]
squared_errors = np.sum((compressed_train_faces - mask)**2, axis = 1)
minimum_error_face = np.argmin(squared_errors)
most_resembling = list(np.where(squared_errors < 20)[0])
#print('Best resembling face in train test: %i' % train_answers[minimum_error_face])

'''
Attempt to get the most similar faces
'''

for k, m in enumerate(most_resembling[:3]):
    plt.subplot(2, 2, 2+k)
    plt.title('Match in train set no. '+str(m))
    plt.axis('off')
    plt.imshow(train_faces[m].reshape(64, 64), cmap=plt.cm.gray, interpolation='nearest')

plt.show()
