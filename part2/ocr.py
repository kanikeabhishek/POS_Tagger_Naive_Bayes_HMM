#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png bc.train txt test-image-file.png
# 
# Authors: Abhishek Kanike(abkanike), Preetham Kowshik(pkowshik), Chuhua Wang(cw234)
# (based on skeleton code by D. Crandall, Oct 2017)
#
#########
# Report:
# File 'bc.train' from part1 is used as training data for ocr. We can actually t
# We first training the data the get the initial probability P(Letter), which is the probability that a sentence
# starts with a certain character (total 72 character). Then it was formed into a dictionary in the format
# of {'A': -log(prob), 'B': -log(prob).....}.
#
# The transition probability P(Letter_t+1|Letter_t) is calculated using the training data. The function
# calculate_transition_probability() returns a dictionary of dictionaries:
# {'A':{'A':-log(prob), 'B':-log(prob),...},'B':{'A':-log(prob), 'B':-log(prob),...}}
#
# Calculating emission probability P(Image|Letter) is slightly complicated than Part1. Each Image is represented as
# a 25*14 matrix, so we first need to calculate the probability for each pixel is black(*) P(Image_pixel|Letter).
# After P(Image_pixel|Letter) is calculated, we insert the testing image and multiply each of them
# to get the P(Image|Letter).
# P(Image|Letter) = Prod_i:350 P(Image_pixel_i|Letter)
# The emission probability is dictionary of dictionaries: {Letter:{Image:prob...}...}
#
# The simplified Bayes net is easy, same as part1.
# P(Letter|Image) = P(Image|Letter) * P(Letter) / P(Image))
# letter'[i] = arg min -log(P (Letter[i]  = letter[i]| Image)).
# Denominator of prior probability is ignored since it will be same for each Image.
# Note, one additional condition is added when black pixel is less than 8, a -10 weight will be assigned to blank space.
# The Variable Elimination is similar to part1, while only forward algorithm is used.
# The Viterbi algorithm calculated the posterior using Prod P(Image_pixel_i|Letter)
# was implemented using the pseudocode on https://en.wikipedia.org/wiki/Viterbi_algorithm.
#
#
# Sample Results:

# ./ocr.py" courier-train.png bc.train test-2-0.png
# Simple: Nos! 14-556! Argued April 28, 2015 - Decided June 26, 2015
# HMM VE: Nos. 14-556! Argued April 28, 2015 - Decided June 26, 2015
# HMM MAP: Nos. 14-556. Argued April 28, 2015 - Dedided June 26, 2015

# ./ocr.py" courier-train.png bc.train test-6-0.png
# Simple: As sbme bf the petitibners in these cases demonstrate, marriage
# HMM VE: As some bf the petitibners in these cases demonstrate, marriage
# HMM MAP: As some of the petitioners in these cases demonstratey marriage

# ./ocr.py" courier-train.png bc.train test-17-0.png
# Simple: It is so ordered.
# HMM VE: Tt is so ordered.
# HMM MAP: It is so ordered.
from __future__ import division
from PIL import Image, ImageDraw, ImageFont
import math
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
# all letters that we assume exists in the training data
LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print im.size
    #print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else '@' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result


def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { LETTERS[i]: letter_images[i] for i in range(0, len(LETTERS) ) }


def read_train_file(fname):
    """
    read train file
    :param fname: train data file name
    :return: list of strings
    """
    return open(fname).readlines()


def calculate_initial_prob(train_file):
    """
    calculate initial probability of the the sentence starts with an alphabet
    :param train_file
    :return: dict
    """
    initial_counts = {LETTERS[i]: 0 for i in range(0, len(LETTERS))}
    for s in train_file:
        if s[0] in LETTERS:
            initial_counts[s[0]] +=1
    return {k: -math.log((v+0.00001) / (total+0.00001)) for total in (sum(initial_counts.itervalues(), 0.0),) for k, v in initial_counts.iteritems()}


def calculate_transition_probability(train_file):
    """
    calculate the transition probability
    :param train_file: Train file name
    :return: dict of dicts
    """
    initial_transition_count= {letter1: {letter2: 0 for letter2 in LETTERS} for letter1 in LETTERS}
    for s in train_file:
        s = s.rstrip('\n')
        for i in range(0,len(s)-1):
            if s[i] in LETTERS:
                if s[i+1] in LETTERS:
                    initial_transition_count[s[i]][s[i+1]] +=1
                    # prevent zero division
    return {l:{k: -math.log((v+0.00001) / (total+0.00001)) for total in (sum(initial_transition_count[l].itervalues(), 0.0),)
              for k, v in initial_transition_count[l].iteritems()} for l in initial_transition_count}


#
def calculate_emission():
    """
    Put the emission probability in to format {Letter:{Image:prob...}...}
    :return: dict of dicts {Letter:{Image:prob...}...}
    """
    emi = {}
    for l in LETTERS:
        emi[l] = {}
        for n in range(len(test_letters)):
            emi[l][n]=prob[n][l]
    return emi


def train_emission():
    """
    Get P(Image_pixel|Letter)
    :return: list of lists of dicts
    """
    black_e=[]

    for i in range(0,25):
        black_e.append([])
        for j in range(0,14):
            temp = {LETTERS[i]: 0 for i in range(0, len(LETTERS))}
            for letter in LETTERS:

                if train_letters[letter][i][j] == '*':
                    temp[letter] += 1
            black_e[i].append(temp)

    for i in range(0,25):
        for j in range(0,14):
            black_e[i][j] = {k: -math.log((v+0.00001) / (total+0.00001)) for total in (sum(black_e[i][j].itervalues(), 0.0),) for k, v in
                               black_e[i][j].iteritems()}
    return black_e


def simplified():
    """
    Simplified classification
    :return: recognized string and a probability list ({Image:{Letter:prob...}...})
    """
    prob=[]
    recog = ''

    for n in range(0,len(test_letters)):
        prob.append({letter1: 1 for letter1 in LETTERS})
        count = 0
        for i in range(0, 25):
            for j in range(0, 14):
                if test_letters[n][i][j] == '*':
                    for le in LETTERS:
                        prob[n][le] += emission_prob[i][j][le]
                        # if le ==' ':
                        #     prob[n][le] = 0
                elif test_letters[n][i][j] == '@':
                    count +=1

        # blank space
        if count > 342:
            prob[n][le] *= -10
            recog += ' '
        else:
            recog += min(prob[0],key=prob[n].get)
    print 'Simple: ' + recog
    return prob


def hmm_ve():
    """
    Variable elimination
    :return: recognized string
    """
    recog = []
    tau_prev = {}
    for counter, image_num in enumerate(image):
        tau_curr = {}
        for le in LETTERS:
            if counter == 0:
                prev_tau_sum = initial_prob[le]
            else:
                prev_tau=[]
                for l in LETTERS:
                    prev_tau.append(tau_prev[l]+transition_prob[l][le])
                prev_tau_sum = sum(prev_tau)
            tau_curr[le] = emi[le][image_num] * prev_tau_sum
        recog.append(tau_curr)
        tau_prev = tau_curr
    print 'HMM VE: ' + ''.join(min(recog[i],key=recog[i].get) for i in range(len(recog)))


# The Viterbi Decoding code was implemented base on the pseudocode on
# https://en.wikipedia.org/wiki/Viterbi_algorithm
def hmm_viterbi():
    """
    Viterbi Decoding using dynamic programming
    :return: recognized string
    """
    V = [{}]
    # base case
    for le in LETTERS:
        V[0][le] = {"log": initial_prob[le] + emi[le][image[0]], "prev": None}
    for t in range(1, len(image)):
        V.append({})
        for le in LETTERS:
            log_temp =[]
            for prev_letter in LETTERS:
                log_temp.append(V[t-1][prev_letter]["log"] + transition_prob[prev_letter][le])
            min_le_log = min(log_temp)
            for prev_letter in LETTERS:
                if V[t-1][prev_letter]["log"] + transition_prob[prev_letter][le] == min_le_log:
                    min_log = min_le_log + emi[le][image[t]]
                    V[t][le] = {"log": min_log, "prev": prev_letter}
                    break
    recog = []
    log=[]
    for value in V[-1].values():
        log.append(value["log"])
    min_log = min(log)
    previous = None

    for le, data in V[-1].items():
        if data["log"] == min_log:
            recog.append(le)
            previous = le
            break

    for t in range(len(V) - 2, -1, -1):
        recog.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    print 'HMM MAP: '+''.join(recog)


#####
# main program

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_txt = read_train_file(train_txt_fname)
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
image = [c for c in range(len(test_letters))]

# get initial probability
initial_prob = calculate_initial_prob(train_txt)
# get transition probability
transition_prob = calculate_transition_probability(train_txt)
# Get P(Image_pixel|Letter) in order to calculate P(Image|Letter)
emission_prob = train_emission()

# Simplified classification
# calculate the probability while do the simplified classification
prob = simplified()
# put the emission probability in to format {Letter:{Image:prob...}...}
emi = calculate_emission()
# Variable Elimination
hmm_ve()
# Viterbi Decoding
hmm_viterbi()
