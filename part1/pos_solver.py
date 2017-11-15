#!/usr/bin/env python
###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
'''
Train data:
Three probabilities are calculated from the train data to predict part of speech(POS) of a sentence
1.  Prior Probability ( Probability of each part of speech from all given words ) -
    A dictionary ( hash os pos ) is used to Prior probability of pos. ["P_pos"]
    Initially train data is parsed line by line. Each line consists of (sentence, pos_bag) pair.
    From pos_bag of sentence, each word's pos counter is incremented by indexing "P_pos".
    After parsing all data, each counter of pos is divided by total number of words.

2.  Transition Probability ( Probability of transition from previous pos to next pos ) -
    A dictionary of dictionary ( hash of hash of pos given pos) is used to store transition probability ["P_Transition"]
    Using train data, from every second word its respective pos is indexed as first dimension in "P_transition",
    previous word pos is indexed as second dimension. Number of such pair is maintained as a counter in "P_transition"
    After parsing all data, each counter value is divided by all possible transition from respective pos.

3.  Emission probablity ( Probablity of given word belongs to a particular pos ) -
    A dictionary of dictionary ( hash of hash of word given pos ) is used to store the emission probabilities ["P_word_pos"]
    From train data, each word and its respective pos is maintained as a counter in "P_word_pos" with pos indexed as
    first dimension and the word as second dimension
    After all data is parsed, each word's pos is divided by the word appeared as different pos

Posterior Probability:
    We use the model 1a irrespective of the algorithm(as discussed in class) inorder to calculate the posterior probabilities.
    Bayes law(applied on Bayes net 1a) is used to calculate the posterior probability given setence and pos of the respective words. The log the posterior probability
    is returned.

Simplified Model:
    Part of Speech tagging of sentence using simplified follows a simple Naive Bayes expansion
                                P(S|W) = P(W|S) * P(S) / P(W)
    This can be generalized for all pos and find the maximum pos probability for each word.
                             s'[i] = arg max P (S[i]  = s[i] |W ).
    For every first word of a sentence, maximum likelihood of the word over all pos multiplied by corresponding intital
    probability of the pos
    Subsequent words pos is estimated by finding maximum likelihood of the word over all pos and multiplied by corresponding
    prior probability of the pos.
    Denominator of prior probability of sentence is ignored since it will be same for each word.

Variable Elimination:
    In above method ( simplified ) dependency of word in predicting pos of every other word is not handled.
    Hence, solving the question by adding above dependency of words in Bayes Net will prediciting pos of sentence better.
    A tau_table with size equal to number of words in a sentence is created.
    For each word two dictionaries are created. First dictionary uses the concept of forward elimination where given word
    is marginalized overall previous words on pos including current word all pos. Second dictionary stores follows the concept of
    backward elimination where a word is marginalized overall future words on pos including current word all pos.
    At the end for for every word, maximum product of forward eliminated probabilities and backward eliminated probablities is
    assigned.
    We need to estimate:
                            P(yi|x1, ..., xT ) = P(x1, x2, ..., xT, y1, y2, .. , yi, .., yT )
    Using Bayes Net we can write all factors as ( estimating a words pos (yi is word[i] and x[i] is respective pos)),
                    P(yi|x1, ..., xT ) = P(y1) * P(x1|y1) * P(y2|y1) * P(x2|y2)... * P(yT |yT ) * P(xT|yT )
    Forward Elimination:
        First word is marginalized overall pos and stored in tau_table as:
                                          tau_table[x[1]][0] = P(y[1]) * P(x[1] | y[1]) * P(y[2]|y[1])
        Similarily, following words are marginalized alongwith previos tau_table values (as a result all previous words will
        be marginalized with current word )
                                  tau_table[x[i]][0] = P(x[i]|y[i]) * P(y[i]|y[i-1]) * tau_table(x[i-1])[0]
    Backward Elimination:
        Last word in marginalized and stored in last index of tau table
                                          tau_table[x1][0] = P(x[T] | y[T])P(y[T] | y[T-1])
        Similarily, all previos words are marginalized alongwith future tau_table values (as a result all future words
        will be marginalized with current word )
                                tau_table[x[i]][1] = P(x[i] | y[i]) * P(y[i+1] | y[i]) * tau_table(x[i+1])[1]
    Finally for each word, maximum from every pos with forward and backward eliminated values are multiplied and assigned.

Viterbi Algorithm:
    This algorithm is used to find the most likely sequence of part-of-speech tagging for a given sentence.
    The algorithm starts off in the forward manner. In the first step the probabilities for all part-of-speech is calculated
    by multiplying the initial probabilities with the emission probabilities for the first word and in the subsequent steps
    the trasition probabilities, emission probabilities and probabilities calculated from the previous step are multiplied and stored in
    t_prob dictionary which has current word position as their keys and probabilities of each pos as their value. Added to that the viterbi_dp_table is used
    to record the pos which gave the maximum probability in the previous step. This is dictionary which consists of pos in the current step as their keys
    and pos which produced the maximum probability for the particular pos  from the previous step as its value.

    Back-tracking:
      After iterating the entire sentence we have got the pos which produced the maximum probabilities in viterbi_dp_table.We begin backtracking by finiding
      the pos which produced the maximum probability is calculated using t_prob and the pos which gave the maximum probability is added to the sequence.
      The key value of the pos which has produced the maximum probability in the previous step is searched in viterbi_dp_table with current
      pos as the value. This process is done until we reach the second word in the sentence. Thus giving us the most likely sequence.

       The mathematical formulation is given below

        Probability of the most likely path ending at state j at a step t+1 is given by

       vj(t+1) = emission_probj(Obv at step t +1) * max [vi(t) trasition_prob[i][j]] for all i in pos


Assumption:
    For every new word ( not present in train data ) from test data, probability of word given pos is assigned with 0.0000001
    value, also if a word is not present in any pos, a smoothing value of 0.0000001 is assigned so that, a near estimate to
    a particular pos will be determined

Results:
    With above model following are the accuracies achieved on given test data ( bc.test ):
    ==> So far scored 2000 sentences with 29442 words.
                       Words correct:     Sentences correct:
       0. Ground truth:      100.00%              100.00%
         1. Simplified:       93.62%               45.00%
             2. HMM VE:       95.21%               55.20%
            3. HMM MAP:       95.18%               55.30%
    When initial probability is total is not assumed for the first word of every sentence following results are observed.
    ==> So far scored 2000 sentences with 29442 words.
                       Words correct:     Sentences correct:
       0. Ground truth:      100.00%              100.00%
         1. Simplified:       93.92%               47.45%
             2. HMM VE:       95.24%               55.40%
            3. HMM MAP:       95.22%               55.50%
'''
####

import math
from collections import Counter

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # All possible part of speech
    ALL_POS = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
    # Emission probablity
    P_word_pos = {}
    # Prior Probability
    P_pos = {}
    # Transition probablity
    P_transition = {}
    # Intital Probability of pos
    P_initial = {}

    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling
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

    # Initialize transition probability
    #

    def initialilze_transition_probability(self):
        return {pos1: {pos2: 0 for pos2 in Solver.ALL_POS} for pos1 in Solver.ALL_POS}

    # Do the training!
    #

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


    # Functions for each algorithm.
    #

    def simplified(self, sentence):
        sentence_pos = []
        for counter, word in enumerate(sentence):
            (max_prob, max_pos) = (0, "")
            if counter == 0:
                for cur_pos in Solver.ALL_POS:
                    if word in self.P_word_pos[cur_pos]:
                        P_Factors = self.P_word_pos[cur_pos][word] * self.P_initial[cur_pos]
                    else:
                        P_Factors = 0.0000001 * self.P_initial[cur_pos]
                    (max_prob, max_pos) = max((max_prob, max_pos), (P_Factors, cur_pos))
            else:
                for cur_pos in Solver.ALL_POS:
                    if word in self.P_word_pos[cur_pos]:
                        P_Factors = self.P_word_pos[cur_pos][word] * self.P_pos[cur_pos]
                    else:
                        P_Factors = 0.0000001 * self.P_pos[cur_pos]
                    (max_prob, max_pos) = max((max_prob, max_pos), (P_Factors, cur_pos))
            sentence_pos.append(max_pos)
        return sentence_pos

    # Variable Elimination
    #

    def hmm_ve(self, sentence):
        sentence_pos = []
        tau_table = []

        # Forward Elimination
        for counter, word in enumerate(sentence):
            tau_table.append([])
            tau_table[-1].append({})
            if counter == 0:
                for cur_pos in Solver.ALL_POS:
                    if word in self.P_word_pos[cur_pos]:
                        P_Factors = self.P_word_pos[cur_pos][word] * self.P_initial[cur_pos]
                    else:
                        P_Factors = 0.0000001 * self.P_initial[cur_pos]
                    tau_table[-1][-1][cur_pos] = P_Factors
            else:
                for cur_pos in Solver.ALL_POS:
                    P_Factors = 0.0
                    if word in self.P_word_pos[cur_pos]:
                        for prev_pos in Solver.ALL_POS:
                            P_Factors += self.P_transition[cur_pos][prev_pos] \
                                         * self.P_word_pos[cur_pos][word] \
                                         * tau_table[-2][-1][prev_pos]
                    else:
                        for prev_pos in Solver.ALL_POS:
                            P_Factors += self.P_transition[cur_pos][prev_pos] \
                                         * 0.0000001 \
                                         * tau_table[-2][-1][prev_pos]
                    tau_table[-1][-1][cur_pos] = P_Factors

        # Backward Elimination
        for counter, word in enumerate(reversed(sentence), 1):
            tau_table[-counter].append({})
            if counter == 1:
                for prev_pos in Solver.ALL_POS:
                    P_Factors = 0.0
                    for cur_pos in Solver.ALL_POS:
                        if word in self.P_word_pos[cur_pos]:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * self.P_word_pos[cur_pos][word]
                        else:
                            P_Factors += self.P_transition[cur_pos][prev_pos] * 0.0000001
                    tau_table[-counter][-1][prev_pos] = P_Factors
            else:
                for prev_pos in Solver.ALL_POS:
                    P_Factors = 0.0
                    for cur_pos in Solver.ALL_POS:
                        if word in self.P_word_pos[cur_pos]:
                            P_Factors += self.P_transition[cur_pos][prev_pos] \
                                         * self.P_word_pos[cur_pos][word] \
                                         * tau_table[-counter+1][-1][cur_pos]
                        else:
                            P_Factors += self.P_transition[cur_pos][prev_pos] \
                                         * 0.0000001 \
                                         * tau_table[-counter+1][-1][cur_pos]
                    tau_table[-counter][-1][prev_pos] = P_Factors

        # Calculate Maximum probability of each word belonging to a POS, independent of other words
        for word in range(len(tau_table)):
            (max_prob, max_pos) = (0, "")
            for pos in Solver.ALL_POS:
                if word < len(tau_table) - 1:
                    (max_prob, max_pos) = max((max_prob, max_pos), ((tau_table[word][0][pos] * tau_table[word+1][1][pos]), pos))
                else:
                    (max_prob, max_pos) = max((max_prob, max_pos), (tau_table[word][0][pos], pos))
            sentence_pos.append(max_pos)

        return sentence_pos

    # Viterbi Algorithm
    #

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
