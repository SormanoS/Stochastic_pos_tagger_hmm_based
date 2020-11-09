import copy
import re
import string
from posTagger.Constants import *


class DataOrganizer:

    @staticmethod
    def sum_word_frequencies(vocabulary_word: dict):
        sum = 0
        for tag in vocabulary_word.keys():
            sum += vocabulary_word[tag]
        return sum

    @staticmethod
    def simple_storing(vocabulary, word, tag):
        """
        store word.lower() in vocabulary

        :param vocabulary: vocabulary of words
        :param word: encountered word
        :param tag: pos tag
        :return: vocabulary
        """
        if vocabulary.keys().__contains__(word):
            if vocabulary[word].keys().__contains__(tag):
                vocabulary[word][tag] += 1
            else:
                vocabulary[word][tag] = 1
        else:
            vocabulary[word] = {tag: 1}
        return vocabulary

    @staticmethod
    def exaustive_storing(vocabulary, word, tag):
        """
        store word.lower(), word.lower.capitalize(), word, word.upper()

        :param vocabulary: vocabulary of words
        :param word: encountered word
        :param tag: pos tag
        :return: vocabulary
        """
        if vocabulary.keys().__contains__(word):
            if vocabulary[word].keys().__contains__(tag):
                if tag == PUNCT or string.punctuation.__contains__(word) or tag == SYM or \
                        (tag == NUM and bool(re.search(r'\d', word))):
                    vocabulary[word][tag] += 1
                else:
                    vocabulary[word.upper()][tag] += 1
                    vocabulary[word.lower()][tag] += 1
                    vocabulary[word.lower().capitalize()][tag] += 1
            else:
                if tag == PUNCT or string.punctuation.__contains__(word) or tag == SYM or \
                        (tag == NUM and bool(re.search(r'\d', word))):
                    vocabulary[word][tag] = 1
                else:
                    vocabulary[word.upper()][tag] = 1
                    vocabulary[word.lower()][tag] = 1
                    vocabulary[word.lower().capitalize()][tag] = 1
        else:
            if tag == PUNCT or string.punctuation.__contains__(word) or tag == SYM or \
                    (tag == NUM and bool(re.search(r'\d', word))):
                vocabulary[word] = {tag: 1}
            else:
                vocabulary[word.upper()] = {tag: 1}
                vocabulary[word.lower()] = {tag: 1}
                vocabulary[word.lower().capitalize()] = {tag: 1}
        return vocabulary

    #
    @staticmethod
    def update_tag_frequencies(pos_frequencies, word, tag, storing_method):
        """
        count tag frequencies

        :param pos_frequencies: tags frequencies dictionary
        :param word: encountered word
        :param tag: pos tag
        :param storing_method: method to store words
        :return:
        """
        if pos_frequencies.keys().__contains__(tag):
            pos_frequencies[tag] += 1
        else:
            pos_frequencies[tag] = 1
        # if storing_method == 3 we need to add 3 frequency of tag for the three
        if storing_method == 3 and not (tag == PUNCT or string.punctuation.__contains__(word) or tag == SYM or
                                        (tag == NUM and bool(re.search(r'\d', word)))):
            pos_frequencies[tag] += 2
        return pos_frequencies

    def build_data_structures(self, training_set, storing_method):
        """
        count word in dataset and build the initial structures

        :param training_set: training set
        :param storing_method: method to store words
        :return: data structures
        """
        # vocabulary contains word -> upos
        vocabulary = dict()
        # couple_pos_count contains word.upos+nextword.upos = count. Ex (Verb follow by Punct = 50) VerbPunct = 50
        couple_pos_count = dict()
        # pos_frequencies
        pos_frequencies = dict()
        # list of tags
        tags = []
        # first tag frequencies
        first_tag_frequencies = dict()
        pre_pos = None
        for sentence in training_set:
            for index, token in enumerate(sentence):
                # build vocabulary word -> (tag1 -> count1), (tag2 -> count2)
                if token.upos is not None:
                    # first tag frequencies. Esample verb -> 3 times first tag
                    if index == 0:
                        first_tag_frequencies = self.update_tag_frequencies(first_tag_frequencies, token.form,
                                                                            token.upos, storing_method)
                    # build list of tags
                    if token.upos not in tags:
                        tags.append(token.upos)
                    if token.form is not None:
                        # storing lower word (parameters: vocabulary, word.lower(), tag)
                        if storing_method == 1:
                            vocabulary = self.simple_storing(vocabulary, token.form.lower(), token.upos)
                        # storing token.form (word encountered in the training set)
                        if storing_method == 2:
                            vocabulary = self.simple_storing(vocabulary, token.form, token.upos)
                    if pre_pos is not None:
                        # count upos frequencies
                        pos_frequencies = self.update_tag_frequencies(pos_frequencies, token.form, token.upos,
                                                                      storing_method)
                        # couple of upos
                        if couple_pos_count.keys().__contains__(pre_pos + token.upos):
                            couple_pos_count[pre_pos + token.upos] += 1
                        else:
                            couple_pos_count[pre_pos + token.upos] = 1
                pre_pos = token.upos
            pre_pos = None
        return vocabulary, couple_pos_count, pos_frequencies, tags, first_tag_frequencies

    def calculates_vocabulary_probabilities(self, vocabulary: dict, pos_frequencies: dict, storing_prob_method: int):
        """
        calculates the probabilities P(w|t).
        storing_prob_type = 2: a word encountered only as ti where ti ∈ I = {NOUN, ADJ, PROPN, ADV, VERB} will
                take ti with 99% probability of vocabulary_pwt[word][pos] and every tj ∈ I \ ti with P = 0.25% of
                vocabulary_pwt[word][pos]

        :param vocabulary: vocabulary of words and their frequencies
        :param pos_frequencies: pos tags frequencies
        :param storing_prob_method: method in wich store probabilities
        :return: vocabularies of probabilities P(w|t)
        """
        vocabulary_pwt = copy.deepcopy(vocabulary)
        vocabulary_ptw = copy.deepcopy(vocabulary)
        for word in vocabulary.keys():
            p = -float('inf')
            for pos in vocabulary[word].keys():
                vocabulary_ptw[word][pos] = vocabulary[word][pos] / self.sum_word_frequencies(vocabulary[word])
                if storing_prob_method == 2 and vocabulary[word].keys().__len__() == 1 and \
                        (pos == NOUN or pos == ADJ or pos == VERB or pos == PROPN or pos == ADV):
                    p = (vocabulary_pwt[word][pos] * 99) / 100
                    vocabulary_pwt[word][pos] = p
                else:
                    vocabulary_pwt[word][pos] = vocabulary[word][pos] / pos_frequencies[pos]
            for pos in pos_frequencies.keys():
                if not vocabulary[word].keys().__contains__(pos):
                    # p != -inf only if storing_prob_type = 2. In that case P(w|t)= 0.25% of P
                    if p != -float('inf') and (pos == NOUN or pos == ADJ or pos == VERB or pos == ADV or
                                               pos == PROPN):
                        vocabulary_pwt[word][pos] = ((p * 100 / 99) - p) / 4
                    else:
                        vocabulary_pwt[word][pos] = -float('inf')
        return vocabulary_pwt, vocabulary_ptw

    @staticmethod
    #
    def calculates_pos_probabilities(couple_pos_count, pos_frequencies, tags, storing_prob_method):
        """
        calculates the conditional probabilities P(t|t-1).
        storing_prob_type = 2: the sum of the tag sequences never encountered, takes on 1% of the sum of the
        probabilities of the sequences encountered. The sum of the tag sequences encountered takes 99% of the
        original probability.
        
        :param couple_pos_count: count of couple of tags
        :param pos_frequencies: pos tags frequencies
        :param tags: list of tags
        :param storing_prob_method: method in wich store the probabilities
        :return: couple_pos_count, conditional probabilities P(t|t-1)
        """
        n_never_ecountered = 0
        sum = 0
        for t in tags:
            for prevt in tags:
                if couple_pos_count.keys().__contains__(prevt + t):
                    couple_pos_count[prevt + t] = couple_pos_count[prevt + t] / pos_frequencies[prevt]
                    if storing_prob_method == 2:
                        sum += couple_pos_count[prevt + t]
                else:
                    couple_pos_count[prevt + t] = -float('inf')
                    n_never_ecountered += 1
            # k is the sequence never encountered. Here we assign sum*1%/n_sequence_never_encountered
        if storing_prob_method == 2:
            p = sum * 99 / 100
            for k in couple_pos_count.keys():
                if couple_pos_count[k] != -float('inf'):
                    # percentage of couple_pos_count [k] on the total * p: respect of the probability distribution
                    couple_pos_count[k] = (couple_pos_count[k] * 100 / sum) * p
                else:
                    couple_pos_count[k] = (sum - p) / n_never_ecountered
        return couple_pos_count

    @staticmethod
    def calculates_first_tag_probabilities1(first_tag_frequencies, tags):
        """
        calculates the probabilities for each tag that it is the first of the phrases (method 1)

        :param first_tag_frequencies: frequencies of tags comparing as the first in a sentence
        :param tags: list of tags
        :return: first tag frequencies dictionary
        """
        tot = sum(first_tag_frequencies.values())
        for t in tags:
            if first_tag_frequencies.keys().__contains__(t):
                first_tag_frequencies[t] = first_tag_frequencies[t] / tot
            else:
                first_tag_frequencies[t] = -float('inf')
        return first_tag_frequencies

    @staticmethod
    def calculates_first_tag_probabilities2(first_tag_probabilities, tags):
        """
        calculates the probabilities for each tag that it is the first of the phrases (method 2)

        :param first_tag_probabilities: frequencies of tags comparing as the first in a sentence
        :param tags: list of tags
        :return: first tag frequencies dictionary
        """
        # 100% of total prob
        tot = sum(first_tag_probabilities.values())
        # number of tag where P = -inf
        null_tags = 17 - first_tag_probabilities.keys().__len__()
        # 99% of total prob
        new_tot = (tot * 99) / 100
        if null_tags != 0:
            for t in tags:
                if first_tag_probabilities.keys().__contains__(t):
                    tag_percentile = (first_tag_probabilities[t] * 100) / tot
                    first_tag_probabilities[t] = (new_tot * tag_percentile) / 100
                else:
                    first_tag_probabilities[t] = (tot - new_tot) / null_tags
        return first_tag_probabilities

    @staticmethod
    def calculate_tag_probabilities(pos_frequencies):
        """
        calculates tag probabilities

        :param pos_frequencies: tags frequencies
        :return: tags probabilities
        """
        tag_probabilities = dict()
        sum_tag = sum(pos_frequencies.values())
        for tag in pos_frequencies.keys():
            tag_probabilities[tag] = pos_frequencies[tag] / sum_tag
        return tag_probabilities

    # calculates the probabilities (word|tag), (tag|tag-1), (tag if the word is the first of the text)
    def calculates_probabilities(self, training_set, storing_method, storing_prob_type):
        """
        calculates the probabilities (word|tag), (tag|tag-1), (tag if the word is the first of the text)

        :param training_set: training_set to analyze
        :param storing_method: method in with words are saved
        :param storing_prob_type: method in with probability is saved. storing_prob_type = 2: if there are n tags that
        have never been met at the beginning of a sentence, they are given a 1% / n probability of the tot, and the
        others are normalized
        :return: analyzed data
        """
        vocabulary, couple_pos_count, pos_frequencies, tags, first_tag_frequencies = \
            self.build_data_structures(training_set, storing_method)
        # calculates the probability distribution for unknown words using the words encountered only once
        unknown_word = self.probability_distribution_for_unknown_words(vocabulary, pos_frequencies)
        vocabulary_pwt, vocabulary_ptw = self.calculates_vocabulary_probabilities(vocabulary, pos_frequencies,
                                                                                  storing_prob_type)
        couple_pos_probabilities = self.calculates_pos_probabilities(couple_pos_count, pos_frequencies, tags,
                                                                     storing_prob_type)
        first_tag_probabilities = self.calculates_first_tag_probabilities1(first_tag_frequencies, tags)

        if storing_prob_type == 2:
            first_tag_probabilities = self.calculates_first_tag_probabilities2(first_tag_probabilities, tags)

        tag_probabilities = self.calculate_tag_probabilities(pos_frequencies)  # frequence_tag/frequence_all_tags
        return vocabulary_pwt, vocabulary_ptw, couple_pos_probabilities, first_tag_probabilities, tags, \
               pos_frequencies, tag_probabilities, unknown_word

    def dictionary_for_unknown_words(self, vocabulary: dict):
        """
        this method builds the probability dictionary for the unknown words

        :param vocabulary: vocabulary of words and frequencies
        :return: unknown words
        """
        unknown_word = dict()
        for word in vocabulary.keys():
            if sum(vocabulary[word].values()) == 1:
                for tag in vocabulary[word].keys():
                    unknown_word = self.simple_storing(unknown_word, UNKNOWN, tag)
        return unknown_word

    def probability_distribution_for_unknown_words(self, vocabulary, pos_frequencies):
        """
        calculates the probability distribution for unknown words using the probability distribution of the words
        encountered only once

        :param vocabulary:
        :param pos_frequencies:
        :return:
        """

        unknown_word = self.dictionary_for_unknown_words(vocabulary)
        for tag in unknown_word[UNKNOWN].keys():
            unknown_word[UNKNOWN][tag] = unknown_word[UNKNOWN][tag] / pos_frequencies[tag]
        return unknown_word

    def __init__(self, training_set, storing_method, storing_prob_method):
        """
            Constructor
            storing_method:
                1: only lower character
                2: case sensitive
                3: word storing using lower, upper and capitalize functions

            storing_prob_type:
                1: word the probability of a word that doesn't appear in the trainig set with a given tag is set to -inf
                2: a word encountered only as NOUN, ADJ or VERB, will take NOUN, ADJ or VERB with 99% probability and
                the other tags with 1% (the probabilities are normalized)

            vocabulary contains word -> (tag1 -> count1), (tag2 -> count2). pwt: P(w|t). ptw: P(t|w) used for baseline
            couple_pos_count contains word.upos+nextword.upos = count. e.g. (Verb follow by Punct = 50) VerbPunct = 50
            couple_word_count contains word+word = count. e.g. (hello follow by my = 20)  hellomy = 20
            pos_frequencies: frequencies of the tags
            tags: list of tags
            first_tag_frequencies: frequencies as the first tag
        """
        self.vocabulary_pwt, self.vocabulary_ptw, self.couple_pos_probabilities, self.first_tag_probabilities, \
        self.tags, self.pos_frequencies, self.tag_probabilities, self.unknown_words_probabilities = \
            self.calculates_probabilities(training_set, storing_method, storing_prob_method)
