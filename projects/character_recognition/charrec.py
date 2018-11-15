# Importing the required modules for the project

import numpy as np # For array manipulation of the images
import matplotlib.pyplot as plt # To plot the graph for presentation and analysis
from PIL import Image # To open and read images
from statistics import mean # To find mean of the array
from collections import Counter # To track the similar match matrix from dataset
import os # To manipulate files


def threshold(image_array):
    balance_array = []
    new_array = image_array.copy()
    
    for each_row in image_array:
        for each_pix in each_row:
            average_num = mean(each_pix[:3])
            balance_array.append(average_num)
            
    balance_num = mean(balance_array)
    
    for each_row in new_array:
        for each_pix in each_row:
            pass
            
            if mean(each_pix[:3]) > balance_num:
                each_pix[0] = 255
                each_pix[1] = 255
                each_pix[2] = 255
                each_pix[3] = 255
            else:
                each_pix[0] = 0
                each_pix[1] = 0
                each_pix[2] = 0
                each_pix[3] = 255
            
                
    return new_array


# Function to generate the training dataset from the number images
def createNumberExamples():
    file_name = 'examples/number_array_examples.txt'
    
    # Remove the already existing file
    if os.path.exists(file_name):
        os.remove(file_name)
    
    number_array_examples = open(file_name, 'a')
    all_numbers = range(0, 10)
    all_versions = range(1, 10)
    
    for each_num in all_numbers:
        for each_version in all_versions:
            # print(str(each_num) + '.' + str(each_version))
            image_file_path = 'images/numbers/' + str(each_num) + '.' + str(each_version) + '.png'
            example_image = Image.open(image_file_path)
            example_image_array = np.asarray(example_image)
            example_image_array_list = str(example_image_array.tolist())
            
            line_to_write = str(each_num) + '::' + example_image_array_list + '\n'
            number_array_examples.write(line_to_write)
            
createNumberExamples()


# Function to generate the training dataset from the alphabet images
def createAlphabetExamples():
    file_name = 'examples/alphabet_array_examples.txt'
    
    # Remove the already existing file
    if os.path.exists(file_name):
        os.remove(file_name)
        
    alphabet_array_examples = open(file_name, 'a')
    all_alphabets = range(0, 10)
    all_versions = range(1, 10)
    
    for each_alpha in all_alphabets:
        for each_version in all_versions:
            # print(str(each_alpha) + '.' + str(each_version))
            image_file_path = 'images/alphabets/' + str(each_alpha) + '.' + str(each_version) + '.png'
            example_image = Image.open(image_file_path)
            example_image_array = np.asarray(example_image)
            example_image_array_list = str(example_image_array.tolist())
            
            line_to_write = str(each_alpha) + '::' + example_image_array_list + '\n'
            alphabet_array_examples.write(line_to_write)
            
createAlphabetExamples()


# Function to generate the training dataset from the devnagri images
def createDevnagriExamples():
    file_name = 'examples/devnagri_array_examples.txt'
    
    # Remove the already existing file
    if os.path.exists(file_name):
        os.remove(file_name)
        
    devnagri_array_examples = open(file_name, 'a')
    all_chars = range(0, 10)
    all_versions = range(1, 10)
    
    for each_char in all_chars:
        for each_version in all_versions:
            # print(str(each_char) + '.' + str(each_version))
            image_file_path = 'images/devnagri/' + str(each_char) + '.' + str(each_version) + '.png'
            example_image = Image.open(image_file_path)
            example_image_array = np.asarray(example_image)
            example_image_array_list = str(example_image_array.tolist())
            
            line_to_write = str(each_char) + '::' + example_image_array_list + '\n'
            devnagri_array_examples.write(line_to_write)
            
createDevnagriExamples()


# Function to guess the probability of a given image to be a prticular image
def guessTheNumber(file_path):
    matched_array = []
    load_examples = open('examples/number_array_examples.txt', 'r').read()
    load_examples = load_examples.split('\n')
    
    # Load and read the given image in a 3D array matrix
    image = Image.open(file_path)
    image_array = np.asarray(image)
    image_array_list = image_array.tolist()
    
    test_sample = str(image_array_list)
    
    for each_example in load_examples:
        if len(each_example) > 3:
            split_example = each_example.split('::')
            current_num = split_example[0]
            current_array = split_example[1]
            
            each_example_pixel = current_array.split('],')
            each_sample_pixel = test_sample.split('],')
            
            x = 0
            while x < len(each_example_pixel):
                if each_example_pixel[x] == each_sample_pixel[x]:
                    matched_array.append(int(current_num))
                    
                x += 1
                
    # print(matched_array)
    x = Counter(matched_array)
    plotResult(x, image_array)
    
    
# Function to guess the probability of a given image to be a prticular image
def guessTheAlphabet(file_path):
    matched_array = []
    load_examples = open('examples/alphabet_array_examples.txt', 'r').read()
    load_examples = load_examples.split('\n')
    
    # Load and read the given image in a 3D array matrix
    image = Image.open(file_path)
    image_array = np.asarray(image)
    image_array_list = image_array.tolist()
    
    test_sample = str(image_array_list)
    
    for each_example in load_examples:
        if len(each_example) > 3:
            split_example = each_example.split('::')
            current_num = split_example[0]
            current_array = split_example[1]
            
            each_example_pixel = current_array.split('],')
            each_sample_pixel = test_sample.split('],')
            
            x = 0
            while x < len(each_example_pixel):
                if each_example_pixel[x] == each_sample_pixel[x]:
                    matched_array.append(int(current_num))
                    
                x += 1
                
    # print(matched_array)
    x = Counter(matched_array)
    # print(x)
    plotResult(x, image_array)
    
    
# Function to guess the probability of a given image to be a prticular image
def guessTheDevnagri(file_path):
    matched_array = []
    load_examples = open('examples/devnagri_array_examples.txt', 'r').read()
    load_examples = load_examples.split('\n')
    
    # Load and read the given image in a 3D array matrix
    image = Image.open(file_path)
    image_array = np.asarray(image)
    image_array_list = image_array.tolist()
    
    test_sample = str(image_array_list)
    
    for each_example in load_examples:
        if len(each_example) > 3:
            split_example = each_example.split('::')
            current_num = split_example[0]
            current_array = split_example[1]
            
            each_example_pixel = current_array.split('],')
            each_sample_pixel = test_sample.split('],')
            
            x = 0
            while x < len(each_example_pixel):
                if each_example_pixel[x] == each_sample_pixel[x]:
                    matched_array.append(int(current_num))
                    
                x += 1
                
    # print(matched_array)
    x = Counter(matched_array)
    # print(x)
    plotResult(x, image_array)
    
    
# Function to plot the given test number and the relative estimates on graph
def plotResult(x, image_array):
    graph_x = []
    graph_y = []
    
    for each_key in x:
        # print(each_key)
        graph_x.append(each_key)
        # print(x[each_key])
        graph_y.append(x[each_key])
        
    fig = plt.figure()
    ax_1 = plt.subplot2grid((4, 4), (0, 0), rowspan = 1, colspan = 4)
    ax_2 = plt.subplot2grid((4, 4), (1, 0), rowspan = 3, colspan = 4)
    
    ax_1.imshow(image_array)
    ax_2.bar(graph_x, graph_y, align = 'center')
    # plt.ylim(400)
    
    xloc = plt.MaxNLocator(12)
    
    ax_2.xaxis.set_major_locator(xloc)
    
    plt.show()

    
# Guessing the number
guessTheNumber('images/numbers/0.7.png')

# Guessing the alphabet
guessTheAlphabet('images/alphabets/0.2.png')

# Guessing the devnagri
guessTheAlphabet('images/devnagri/2.2.png')