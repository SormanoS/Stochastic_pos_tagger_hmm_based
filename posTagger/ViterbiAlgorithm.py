import math

import numpy as np

from posTagger.DataOrganizer import DataOrganizer
from posTagger.Constants import *


class ViterbiAlgorithm:
    """
    Constructor
    smoothing_type:
        1: unknown word = PROPN
        2: unknown word = probability distribution of all tags (frequency_tag/frequency_all_tag)
        3: unknown word tag follow the probability distribution of the words encountered only once
    """

    def __init__(self, data: DataOrganizer, smoothing_type):
        self.smoothing_type = smoothing_type
        self.data: DataOrganizer = data
        self.pos_tagging = []

    def calculate_first_column(self, matrix, word):
        """
        first matrix column: prior probability of tag P(tag is first) * P(word|tag)

        :param matrix: probability matrix
        :param word: encountered word
        :return: probability matrix
        """
        maximum = -float('inf')
        word_tag = ""
        for tindex in range(self.data.tags.__len__()):
            # known word
            if self.data.vocabulary_pwt.keys().__contains__(word):
                if self.data.vocabulary_pwt[word][self.data.tags[tindex]] != 0 and \
                        self.data.vocabulary_pwt[word][self.data.tags[tindex]] != -float('inf') and \
                        self.data.first_tag_probabilities[self.data.tags[tindex]] != 0 and \
                        self.data.first_tag_probabilities[self.data.tags[tindex]] != -float('inf'):
                    matrix[tindex][0] = math.log10(self.data.vocabulary_pwt[word][self.data.tags[tindex]]) + \
                                        math.log10(self.data.first_tag_probabilities[self.data.tags[tindex]])
                else:
                    matrix[tindex][0] = -float('inf')
            else:
                # unknown word = PROPN
                if self.smoothing_type == 1:
                    if self.data.tags[tindex] == PROPN:
                        matrix[tindex][0] = 1
                    else:
                        matrix[tindex][0] = -float('inf')
                # unknown word tag follow the frequency of tags
                elif self.smoothing_type == 2:
                    matrix[tindex][0] = math.log10(self.data.tag_probabilities[self.data.tags[tindex]])
                # unknown word tag follow the probability distribution of the words encountered once
                elif self.smoothing_type == 3:
                    if self.data.unknown_words_probabilities[UNKNOWN].keys().__contains__(self.data.tags[tindex]):
                        matrix[tindex][0] = \
                            math.log10(self.data.unknown_words_probabilities[UNKNOWN][self.data.tags[tindex]])
                    else:
                        matrix[tindex][0] = -float('inf')
            if maximum <= matrix[tindex][0]:
                maximum = matrix[tindex][0]
                word_tag = self.data.tags[tindex]

        self.pos_tagging.append(word_tag)

        return matrix

    def calculate_matrix_cell_probabilities(self, matrix, cell_column, tag, word):
        """
        calculate a matrix cell probabilities

        :param matrix: matrix of probabilities
        :param cell_column: column of the cell
        :param tag: pos tag
        :param word: encountered word
        :return: maximum probability
        """
        emission_probability = 0
        maximum = -float('inf')
        # known word
        if self.data.vocabulary_pwt.keys().__contains__(word):
            emission_probability = self.data.vocabulary_pwt[word][tag]
        else:
            # unknown word = PROPN
            if self.smoothing_type == 1:
                if tag == PROPN:
                    emission_probability = 1
                else:
                    emission_probability = -float('inf')
            # unknown word tag follow the frequence of tag
            elif self.smoothing_type == 2:
                emission_probability = self.data.tag_probabilities[tag]
            # unknown word tag follow the probability distribution of the words encountered once
            elif self.smoothing_type == 3:
                if self.data.unknown_words_probabilities[UNKNOWN].keys().__contains__(tag):
                    emission_probability = self.data.unknown_words_probabilities[UNKNOWN][tag]
                else:
                    emission_probability = -float('inf')
        # probability for every tag (every cell)
        for tindex in range(self.data.tags.__len__()):
            if emission_probability == 0 or emission_probability == -float('inf') or \
                    self.data.couple_pos_probabilities[self.data.tags[tindex] + tag] == -float('inf'):
                result = -float('inf')
            else:
                result = math.log10(emission_probability) + \
                         math.log10(self.data.couple_pos_probabilities[self.data.tags[tindex] + tag]) + \
                         matrix[tindex][cell_column - 1]
            if maximum <= result:
                maximum = result

        return maximum

    def predict(self, sequence):
        """
        # predict the tags vector
        :param sequence: sentence
        :return: pos tagging of the sentence
        """
        word_tag = ""
        string = sequence.split(" ")
        # matrix: column -> words of sentence, row -> tags
        matrix = self.calculate_first_column(np.ones((self.data.tags.__len__(), string.__len__())), string[0])
        maximum = -float('inf')
        # columns
        for col in range(1, string.__len__(), 1):
            # rows
            for tindex in range(self.data.tags.__len__()):
                matrix[tindex][col] = self.calculate_matrix_cell_probabilities(matrix, col, self.data.tags[tindex],
                                                                               string[col])
                if maximum <= matrix[tindex][col]:
                    maximum = matrix[tindex][col]
                    word_tag = self.data.tags[tindex]
            maximum = -float('inf')
            self.pos_tagging.append(word_tag)
        return self.pos_tagging
