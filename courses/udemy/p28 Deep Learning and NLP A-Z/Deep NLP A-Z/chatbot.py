# Building a chatbot with deep NLP

# importing the libraries
import numpy as np
import re
import tensorflow as tf
import time

##### PART-1 DATA PREPROCESSING #####

# Importing the dataset
lines = open('movie_lines.txt', encoding="UTF-8", errors="ignore").read().split("\n")
conversations = open('movie_conversations.txt', encoding="UTF-8", errors="ignore").read().split("\n")

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line)==5:
        id2line[_line[0]] = _line[4]

# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))

# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Creating two dictionaries that map the questions words and the answer word to unique integer
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count > threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count > threshold:
        answerswords2int[word] = word_number
        word_number += 1

# Adding the last token to these dictionaries
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Creating the inverse dictionary of the answerwords2int dictionary
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}

# Adding the EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"

# Translating all the questions and the answers into integers
# and replacing all the words that were filtered out by <OUT>
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int["<OUT>"])
        else:
            ints.append(questionswords2int[word])
    
    questions_to_int.append(ints)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int["<OUT>"])
        else:
            ints.append(answerswords2int[word])
    
    answers_to_int.append(ints)
    
# Sorting questions and answers by the length of the questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


##### Part-2 BUILDING THE SEQ2SEQ MODEL #####

# Creating placeholder for the inputs and the targets
def model_inputs():
    # placeholder for the input (datatype, dimension of the matrix, name assigned by us to the data)
    inputs = tf.placeholder(tf.int32, [None, None], name = "input")
    targets = tf.placeholder(tf.int32, [None, Node], name = "target")
    lr = tf.placeholder(tf.float32, name = "learning_rate")
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    return inputs, targets, lr, keep_prob





