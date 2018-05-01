'''Applying learning on a complex set of images, called the labelled faces in the
wild dataset that contains images of famous people collected over the internet.
http://scikit-learn.org/stable/datasets/labeled_faces.html
'''

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person = 60, resize = 0.4)
X = lfw_people.data
Y = lfw_people.target
target_names = [lfw_people.target_names[a] for a in y]
n_samples, h, w = lfw_people.images.shape

from collections import Counter
#for name, count in Counter(target_names).item():
#    print("%20s %i" % (name, count))


'''
Now dividing the examples into training and test sets, you can display a sample of
pictures from both sets depicting Jun'Ichiro Koizumi, Prime Minister of Japan from
2001 to 2006.
'''

from sklearn.cross_validation import StritifiedShuffleSplit
train, test = list(StratifiedShuffleSplit(target_names, n_iter=1, test_size=0.1, random_state=101))[0]

import matplotlib.pyplot as plt
plt.subplot(1, 4, 1)
plt.axis('off')

for k, m in enumerate(X[train][y[train]]==6][:4]):
    plt.subplot(1, 4, 1+k)
    if k==0:
        plt.title('Train Set')
        
    plt.axis('off')
    plt.imshow(m.reshape(50, 37), cmap=plt.cm.gray, interpolation = 'nearest')

plt.show()

for k, m in enumerate(X[test][y[test]==6][:4]):
    plt.subplot(1, 4, 1+k)
    if k==0:
        plt.title('Test Set')
        
    plt.axis('off')
    plt.imshow(m.reshape(50, 37), cmap=plt.cm.gray, interpolation = 'nearest')

plt.show()
