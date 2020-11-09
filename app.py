from argparse import ArgumentParser, Namespace

import pyconll
import pyconll.util

from posTagger import DataOrganizer as Do
from posTagger.ViterbiAlgorithm import ViterbiAlgorithm
from posTagger.Constants import *


# insert the space between the last two character of the sentence
def build_phrase(sentence, storing_type):
    result = ''
    for index, token in enumerate(sentence):
        if index < sentence.__len__() - 1:
            if storing_type == 1:
                result += token.form.lower() + " "
            else:
                result += token.form + " "
        else:
            if storing_type == 1:
                result += token.form.lower()
            else:
                result += token.form
    return result


def select_max_tag(vocabulary_ptw):
    """
    select the tag with max probability
    """
    maximum = 0
    best_tag = ""
    for tag in vocabulary_ptw.keys():
        if maximum < vocabulary_ptw[tag]:
            maximum = vocabulary_ptw[tag]
            best_tag = tag
    return best_tag


def test_word(viterbi_algorithm, result, index, word, tag, ntoken, correct, correct_baseline, unknown_word):
    """
    test a word
    """
    if tag is not None:
        ntoken += 1
        if tag == result[index]:
            correct += 1
        if viterbi_algorithm.data.vocabulary_ptw.keys().__contains__(word):
            if tag == select_max_tag(viterbi_algorithm.data.vocabulary_ptw[word]):
                correct_baseline += 1
        else:
            if tag == PROPN:
                correct_baseline += 1
            unknown_word += 1
    return ntoken, correct, correct_baseline, unknown_word


def test_sentence(viterbi_algorithm, sentence, result, correct, correct_baseline, ntoken, unknown_word, storing_type):
    """
    test a sentence
    """
    for index, token in enumerate(sentence):
        if storing_type == 1:
            word = token.form.lower()
        else:
            word = token.form
        ntoken, correct, correct_baseline, unknown_word = test_word(viterbi_algorithm, result, index, word, token.upos,
                                                                    ntoken, correct, correct_baseline, unknown_word)
    return correct, correct_baseline, ntoken, unknown_word


def test_accuracy(file_path: str, viterbi_algorithm: ViterbiAlgorithm, storing_method: int):
    """
    calculate the accuracy with viterbi and the baseline
    """
    train = pyconll.load_from_file(file_path)
    ntoken = 0
    correct = 0
    correct_baseline = 0
    unknown_word = 0
    for sentence in train:
        phrase = build_phrase(sentence, storing_method)
        result = viterbi_algorithm.predict(phrase)
        # if test only case without del = di + il
        if sentence.__len__() == result.__len__():
            correct, correct_baseline, ntoken, unknown_word = test_sentence(viterbi_algorithm, sentence, result,
                                                                            correct, correct_baseline, ntoken,
                                                                            unknown_word, storing_method)
        result.clear()
    accuracy = round(correct / ntoken * 100, ndigits=2)
    baseline = round(correct_baseline / ntoken * 100, ndigits=2)
    return accuracy, baseline, unknown_word


def main(args: Namespace):
    print("Configurations: " + str(args))

    training_set = pyconll.load_from_file(args.training_set_path)
    if args.validation_set_path is not None:
        training_set += pyconll.load_from_file(args.validation_set_path)
    data = Do.DataOrganizer(training_set=training_set, storing_method=args.storing_method,
                            storing_prob_method=args.storing_prob_method)
    va = ViterbiAlgorithm(data=data, smoothing_type=args.smoothing_method)
    accuracy, baseline, unknown_word = test_accuracy(file_path=args.test_set_path, viterbi_algorithm=va,
                                                     storing_method=args.storing_method)
    print("baseline: " + str(baseline) + "%")
    print("Pos tagger accuracy: " + str(accuracy) + "%")
    print("unknown words: " + str(unknown_word))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--storing-data', type=int, choices=[1, 2], default=1, dest='storing_method',
                        help='words storing method')
    parser.add_argument('--storing-prob', type=int, choices=[1, 2], default=1, dest='storing_prob_method',
                        help='probability storing method')
    parser.add_argument('--smoothing', type=int, choices=[1, 2, 3], default=1, dest='smoothing_method',
                        help='smoothing method')
    parser.add_argument('--training-set', type=str, required=True, dest='training_set_path',
                        help='training set path')
    parser.add_argument('--validation-set', type=str, dest='validation_set_path',
                        help='validation set path: this dataset will be added to the training set')
    parser.add_argument('--test-set', type=str, required=True, dest='test_set_path',
                        help='test set path')
    main(parser.parse_args())
