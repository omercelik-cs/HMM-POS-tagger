#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-

from random import shuffle

# read the whole content into a list with utf-8 format
def retrieve_content(filename):
    with open(filename, "rb") as fp:
        content = []
        for line in fp:
            content.append(line)
    # remove whitespace at the end of each line
    content = [x.strip().decode('utf-8') for x in content]
    return content

# parse the tokens as word tag pairs as list items (2nd, 4th columns)
def parse_corpus_word_tag(content):
    column_24 = []
    corpus = []
    for line in content:
        # if line is not empty we continue in same sentence
        if (line) != '':
            splitted = line.split('\t')
            # ignore the lines having _ score in 2nd column
            if splitted[1] != '_':
                # get the value in 2nd and 4th columns
                column_24.append((splitted[1], splitted[3]))
        else:
            corpus.append(column_24)
            column_24 = []
    return corpus

# store the frequencies of each tag and each word-tag pair in the corpus
# also consider the start mark (e.g. <s>,Noun)
def compute_word_tag_frequencies(training_corpus):
    word_tag_frequncies = {}
    tag_frequencies = {}
    start_mark = "<s>"
    for sentence in training_corpus:
        for pair in sentence:
            # store word_tag_frequencies in a dictionary
            if pair in word_tag_frequncies:
                word_tag_frequncies[pair] += 1
            else:
                word_tag_frequncies[pair] = 1
            # store tag frequencies in a dictionary
            if pair[1] in tag_frequencies:
                tag_frequencies[pair[1]] += 1
            else:
                tag_frequencies[pair[1]] = 1
        # add the start mark frequency as number of sentences
        tag_frequencies[start_mark] = len(training_corpus)
    return word_tag_frequncies,tag_frequencies

# store the frequencies of each tag-tag pairs (transition) in the corpus
def compute_bigram_tag_frequencies(training_corpus):
    bigram_tag_frequencies = {}
    start_mark = "<s>"
    for sentence in training_corpus:
        # store frequencies of start mark with other tags
        bigram_tag = (start_mark,sentence[0][1])
        if bigram_tag in bigram_tag_frequencies:
            bigram_tag_frequencies[bigram_tag] += 1
        else:
            bigram_tag_frequencies[bigram_tag] = 1
        # store frequencies of tags between each other
        for i in range(1,len(sentence)):
            bigram_tag = (sentence[i - 1][1], sentence[i][1])
            if bigram_tag in bigram_tag_frequencies:
                bigram_tag_frequencies[bigram_tag] += 1
            else:
                bigram_tag_frequencies[bigram_tag] = 1
    return bigram_tag_frequencies

# split the corpus into training and test set as 90%
def split_train_test(corpus):
    shuffle(corpus)
    training_corpus = corpus[: int(len(corpus) * .9)]
    test_corpus = corpus[int(len(corpus) * .9):]
    return training_corpus,test_corpus

# compute emission probabilities for each word-tag pairs no smoothing
def compute_emission_probabilities(word_tag_frequencies,tag_frequencies):

    emission_probs = {}
    for word_tag, frequency in word_tag_frequencies.items():
        emission_probs[word_tag] = (float(frequency)/(tag_frequencies[word_tag[1]]))
    return emission_probs

# compute transition probabilities of bigram POS tags
def compute_transition_probabilities(bigram_tag_frequencies,tag_frequencies):
    transition_probs = {}
    for bigram_tag, frequency in bigram_tag_frequencies.items():
        transition_probs[bigram_tag] = float(frequency)/ tag_frequencies[bigram_tag[0]]
    return transition_probs

def compute_word_tag_set(training_corpus):
    # the key is word and the value is the set of tags
    word_tagset_dictionary = {}
    for sentence in training_corpus:
        for pair in sentence:
            word,tag = pair[0],pair[1]
            if word in word_tagset_dictionary:
                tag_set = word_tagset_dictionary[word]
                tag_set.add(tag)
            else:
                tag_set = set([tag])
                word_tagset_dictionary[word] = tag_set
    return word_tagset_dictionary

if __name__ == "__main__":

    # read and parse the content, create training test data
    filename = "Project (Application 1) (MetuSabanci Treebank).conll"
    content = retrieve_content(filename)
    word_tag_corpus = parse_corpus_word_tag(content)
    training_corpus, test_corpus = split_train_test(word_tag_corpus)

    # get unique tags for each word in the training corpus
    word_tagset_dictionary = compute_word_tag_set(training_corpus)

    # get the frequency of each word-tag pair
    word_tag_frequencies, tag_frequencies= compute_word_tag_frequencies(training_corpus)

    #get the frequency of tag-tag pair (bigram tag)
    bigram_tag_frequencies = compute_bigram_tag_frequencies(training_corpus)

    # learn emission probabilities
    emission_probabilities = compute_emission_probabilities(word_tag_frequencies,tag_frequencies)

    # learn transition probabilities
    transition_probabilities = compute_transition_probabilities(bigram_tag_frequencies,tag_frequencies)

    # store all training parameters in a dictionary to save in a file
    training_parameters = {"word_tag_frequencies": word_tag_frequencies,
                           "tag_frequencies": tag_frequencies,
                           "bigram_tag_frequencies": bigram_tag_frequencies,
                           "word_tagset_dictionary": word_tagset_dictionary,
                           "transition_probabilities": transition_probabilities,
                           "emission_probabilities": emission_probabilities,}