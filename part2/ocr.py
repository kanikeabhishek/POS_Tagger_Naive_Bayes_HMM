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


def hmm_ve(observations, states, start_prob, trans_prob, emm_prob, end_st):
    # forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k]+trans_prob[k][st] for k in states)

            f_curr[st] = emm_prob[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr
    print 'HMM VE: ' + ''.join(min(fwd[i],key=fwd[i].get) for i in range(len(fwd)))




def hmm_viterbi():
    V = [{}]
    for st in LETTERS:
        V[0][st] = {"prob": initial_prob[st] + emi[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in LETTERS:
            max_tr_prob = min(V[t-1][prev_st]["prob"] + transition_prob[prev_st][st] for prev_st in LETTERS)
            for prev_st in LETTERS:
                if V[t-1][prev_st]["prob"] + transition_prob[prev_st][st] == max_tr_prob:
                    min_log = max_tr_prob + emi[st][obs[t]]
                    V[t][st] = {"prob": min_log, "prev": prev_st}
                    break
    opt = []
    # The lowest log
    min_log = min(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == min_log:
            opt.append(st)
            previous = st
            break
     # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print 'HMM MAP: '+''.join(opt)


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_txt = read_train_file(train_txt_fname)
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
obs = [c for c in range(len(test_letters))]

initial_prob = calculate_initial_prob(train_txt)
transition_prob = calculate_transition_probability(train_txt)
emission_prob = train_emission()
prob=simplified()
emi = calculate_emission()
hmm_ve(obs,LETTERS,initial_prob,transition_prob,emi,'End')
hmm_viterbi()