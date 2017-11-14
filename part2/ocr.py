#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)
#
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
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

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
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    initial_counts = {TRAIN_LETTERS[i]: 0 for i in range(0, len(TRAIN_LETTERS))}
    for s in train_file:
        if s[0] in TRAIN_LETTERS:
            initial_counts[s[0]] +=1
    return {k: -math.log((v+0.00001) / (total+0.00001)) for total in (sum(initial_counts.itervalues(), 0.0),) for k, v in initial_counts.iteritems()}


def calculate_transition_probability(train_file):
    """
    calculate the transition probability
    :param train_file: Train file name
    :return:
    """
    initial_transition_count= {letter1: {letter2: 0 for letter2 in LETTERS} for letter1 in LETTERS}
    for le in initial_transition_count:
        initial_transition_count[le]['End'] = 0
    for s in train_file:
        s = s.rstrip('\n')
        for i in range(0,len(s)-1):
            if s[i] in LETTERS:
                if s[i+1] in LETTERS:
                    initial_transition_count[s[i]][s[i+1]] +=1
        if s[-1] in LETTERS:
            initial_transition_count[s[-1]]['End'] +=1

    return {l:{k: -math.log((v+0.00001) / (total+0.00001)) for total in (sum(initial_transition_count[l].itervalues(), 0.0),)
              for k, v in initial_transition_count[l].iteritems()} for l in initial_transition_count}
    # +1 to prevent zero division

#
def calculate_emission():
    emi = {}
    for l in LETTERS:
        emi[l] = {}
        for n in range(len(test_letters)):
            emi[l][n]=prob[n][l]
    return emi


def train_emission():
    """
    Get P(Image_pixel|Letter)
    :return:
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
    :return:
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

        if count > 340:
            prob[n][le] =-10
            recog += ' '
        # elif count == 338:
        #     recog += '.'
        else:
            recog += min(prob[0],key=prob[n].get)
    print 'Simple: ' + recog
    return prob


def hmm_ve():
    """
    Variable elimination
    :return:
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


# For Viterbi, the code was implemented base on the pseudocode in
# https://en.wikipedia.org/wiki/Viterbi_algorithm
def hmm_viterbi():
    """
    Viterbi using dynamic programming
    :return:
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

# calculate the probability while do the simplified classification
prob = simplified()
# put the emission probability in to format {Letter:{Image:prob...}...}
emi = calculate_emission()
hmm_ve()
hmm_viterbi()