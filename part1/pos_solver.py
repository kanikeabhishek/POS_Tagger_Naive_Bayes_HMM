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
from collections import Counter

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    P_word_pos = {}
    P_pos = {}
    P_transition = {}
    ALL_POS = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        self.posterior_prob = 1

        for index in range(len(sentence)):
            word,pos = (sentence[index],label[index])
            if(index == 0):
                if(word in self.P_word_pos[pos]):
                    self.posterior_prob *= self.P_initial[pos] * self.P_word_pos[pos][word]
                else:
                    self.posterior_prob *= self.P_initial[pos] * 0.0000001
            else:
                if(word in self.P_word_pos[pos]):
                    self.posterior_prob *= self.P_transition[pos][label[index-1]] * self.P_word_pos[pos][word]
                else:
                    self.posterior_prob *= self.P_transition[pos][label[index-1]] * 0.0000001

        return math.log(self.posterior_prob) if self.posterior_prob > 0 else 0

    # Do the training!
    #

    def initialilze_transition_probability(self):
        return {pos1: {pos2: 0 for pos2 in self.ALL_POS} for pos1 in self.ALL_POS}


    def train(self, data):
        number_of_words = 0
        # Initialize transition probability
        self.P_transition = self.initialilze_transition_probability()

        # Calculate Initial probabilty
        self.P_initial = Counter([pos_bag[0] for _, pos_bag in data])
        for pos, count in self.P_initial.items():
            self.P_initial[pos] /= float(len(data))

        # Count for prior and conditional probabilities
        for (sentence, pos_bag) in data:
            for item in range(0, len(sentence)):
                if item > 0:
                    self.P_transition[pos_bag[item]][pos_bag[item-1]] += 1
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

        # Calculate Transition probabilities of pos as hash of hash of pos
        for cur_pos in self.P_transition:
            total = sum(self.P_transition[cur_pos].values())
            for next_pos in self.P_transition[cur_pos]:
                self.P_transition[cur_pos][next_pos] /= float(total)

        # Caluclate Emission probabilities of word | pos as hash of pos of word
        for gt_pos in self.P_pos:
            self.P_pos[gt_pos] /= float(number_of_words)
            total_words_pos = sum([count for word, count in self.P_word_pos[gt_pos].items()])
            for word in self.P_word_pos[gt_pos]:
                self.P_word_pos[gt_pos][word] /= float(total_words_pos)
        #print(self.P_word_pos)
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        sentence_pos = []
        for word in sentence:
            (max_prob, max_pos) = (0, "")
            for cur_pos in self.ALL_POS:
                if word in self.P_word_pos[cur_pos]:
                    P_Factors = self.P_word_pos[cur_pos][word] * self.P_pos[cur_pos]
                else:
                    P_Factors = 0.0000001 * self.P_pos[cur_pos]
                (max_prob, max_pos) = max((max_prob, max_pos), (P_Factors, cur_pos))
            sentence_pos.append(max_pos)
        return sentence_pos

    def hmm_ve(self, sentence):
        sentence_pos = []
        tau_table = []

        # Forward Elimination
        for counter, word in enumerate(sentence):
            tau_table.append([])
            tau_table[-1].append({})
            if counter == 0:
                for cur_pos in self.ALL_POS:
                    if word in self.P_word_pos[cur_pos]:
                        P_Factors = self.P_word_pos[cur_pos][word] * self.P_pos[cur_pos]
                    else:
                        P_Factors = 0.0000001 * self.P_pos[cur_pos]
                    tau_table[-1][-1][cur_pos] = P_Factors
            else:
                for cur_pos in self.ALL_POS:
                    P_Factors = 0.0
                    if word in self.P_word_pos[cur_pos]:
                        for prev_pos in self.ALL_POS:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * self.P_word_pos[cur_pos][word] * tau_table[-2][-1][prev_pos]
                    else:
                        for prev_pos in self.ALL_POS:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * 0.0000001 * tau_table[-2][-1][prev_pos]
                    tau_table[-1][-1][cur_pos] = P_Factors

        # Backward Elimination
        for counter, word in enumerate(reversed(sentence), 1):
            tau_table[-counter].append({})
            if counter == 1:
                for prev_pos in self.ALL_POS:
                    for cur_pos in self.ALL_POS:
                        if word in self.P_word_pos[cur_pos]:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * self.P_word_pos[cur_pos][word]
                        else:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * 0.0000001
                    tau_table[-counter][-1][prev_pos] = P_Factors
            else:
                for prev_pos in self.ALL_POS:
                    for cur_pos in self.ALL_POS:
                        if word in self.P_word_pos[cur_pos]:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * self.P_word_pos[cur_pos][word] * tau_table[-counter+1][-1][cur_pos]
                        else:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * 0.0000001 * tau_table[-counter+1][-1][cur_pos]
                    tau_table[-counter][-1][prev_pos] = P_Factors

        # Calculate Maximum probability of each word belonging to a POS, independent of other words
        for word in range(len(tau_table)):
            (max_prob, max_pos) = (0, "")
            for pos in self.ALL_POS:
                if word < len(tau_table) - 1:
                    (max_prob, max_pos) = max((max_prob, max_pos), ((tau_table[word][0][pos] * tau_table[word+1][1][pos]), pos))
                else:
                    (max_prob, max_pos) = max((max_prob, max_pos), (tau_table[word][0][pos], pos))
            sentence_pos.append(max_pos)

        return sentence_pos


    def hmm_viterbi(self, sentence):

        self.t_prob = {}
        self.viterbi_dp_table = { i:dict() for i in range(len(sentence))}

        for i in range(len(sentence)):
            self.t_prob[i] = {}
            word = sentence[i]

            for pos in Solver.ALL_POS:
                self.t_prob[i][pos] = 0

            if(i==0):
                for cur_pos in Solver.ALL_POS:
                    if(word in self.P_word_pos[cur_pos]):
                        self.t_prob[0][cur_pos] = self.P_initial[cur_pos] * self.P_word_pos[cur_pos][word]
                    else:
                        self.t_prob[0][cur_pos] = self.P_initial[cur_pos] * 0.0000001
            else:
                for cur_pos in Solver.ALL_POS:
                    max_probability = 0
                    max_pos = ''
                    for prev_pos in Solver.ALL_POS:
                        if(word in self.P_word_pos[cur_pos]):
                            prob = self.t_prob[i-1][prev_pos]\
                                     * self.P_transition[cur_pos][prev_pos]\
                                     * self.P_word_pos[cur_pos][word]
                        else:
                            prob = self.t_prob[i-1][prev_pos]\
                                     * self.P_transition[cur_pos][prev_pos]\
                                     * 0.0000001

                        if(max_probability < prob):
                            max_probability = prob
                            max_pos = prev_pos

                    self.t_prob[i][cur_pos] = max_probability
                    self.viterbi_dp_table[i][cur_pos] = max_pos

        sequence = []
        sequence.append(max(self.t_prob[len(sentence)-1].iterkeys(),key = lambda k:self.t_prob[len(sentence)-1][k]))
        for i in range(len(sentence)-1,0,-1):
            cur_pos = sequence[-1]
            for key_pos in self.viterbi_dp_table[i].keys():
                if(key_pos == cur_pos):
                    sequence.append(self.viterbi_dp_table[i][key_pos])
        sequence.reverse()
        return sequence
        #return [ "noun" ] * len(sentence)


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
