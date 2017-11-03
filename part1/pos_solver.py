###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    P_word_pos = {}
    P_pos = {}
    ALL_POS = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        number_of_words = 0
        for (sentence, pos_bag) in data:
            for item in range(0, len(sentence)):
                number_of_words += 1
                if pos_bag[item] in self.P_pos:
                    self.P_pos[pos_bag[item]] += 1
                else:
                    self.P_pos[pos_bag[item]] = 1
                    self.P_word_pos[pos_bag[item]] = {}

                if sentence[item] in self.P_word_pos[pos_bag[item]]:
                    self.P_word_pos[pos_bag[item]][sentence[item]] += 1
                else:
                    self.P_word_pos[pos_bag[item]][sentence[item]] = 1

        for gt_pos in self.P_pos:
            self.P_pos[gt_pos] /= float(number_of_words)
            total_words_pos = sum([self.P_word_pos[gt_pos][word] for word in self.P_word_pos[gt_pos]])
            for word in self.P_word_pos[gt_pos]:
                self.P_word_pos[gt_pos][word] /= float(total_words_pos)

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        sentence_pos = []
        for word in sentence:
            (max_prob, max_pos) = (0, '')
            for cur_pos in self.ALL_POS:
                if word in self.P_word_pos[cur_pos]:
                    (max_prob, max_pos) = max((max_prob, max_pos), (self.P_word_pos[cur_pos][word], cur_pos))
            sentence_pos.append(max_pos)
        return sentence_pos

    def hmm_ve(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

