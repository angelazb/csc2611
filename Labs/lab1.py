from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import scipy
from typing import TextIO
from nltk.corpus import brown
from nltk.book import *
import re
import math
from sklearn.decomposition import PCA
import sklearn
import sklearn.metrics

'''
Step 1
'''
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
#print(model['dog'])

'''
Step 2
'''
# I created a csv with the word pairs + a column to indicate if the pair of words is in W or not
rg65 = pd.read_csv('Labs/rg65.csv')
rg65_inW = pd.read_csv('Labs/rg65_inW.csv')

'''
Step 3
'''
# Function to get the cosine distance between two words using w2v model
def cosine_distance_w2v(word1, word2):
    vec1 = model[word1]
    vec2 = model[word2]

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return np.dot(vec1, vec2) / (norm1 * norm2)

n = len(rg65_inW['word1'])

# COSINE DISTANCE
w2v = []
for i in range(n):
    w1 = rg65_inW['word1'][i]
    w2 = rg65_inW['word2'][i]
    w2v.append(cosine_distance_w2v(w1, w2))

# PEARSON CORRELATION
print(w2v)
print("Pearson Correlation w2v and human:", scipy.stats.pearsonr(rg65_inW['value'], w2v))

'''
We observe that the correlation of word2vec and human similarities is extremely higher
compared to the LSA comparisons from the previous exercise. For the words in RG65 that 
are also in W, there is a correlation of 0.919. This shows that the previous models
similarities did not have sufficient training data and doesn't perform as well when the
data is large.
'''

'''
Step 4
'''
analogy_values =  model.wv.evaluate_word_analogies('word-test.txt')

# All words from the brown corpus
# Extract the 5000 most common words denoted by W
words = FreqDist(brown.words())
W_tuples = words.most_common(5300)
W = [t[0].lower() for t in W_tuples if not re.match('.*[^a-z]+', t[0])][:5000]

word_test = open("word-test.txt", "r")
test_semantic = []
test_syntactic = []

line = word_test.readline().strip()
s = 0
while line != "":
    # Check it's a semantic or sysntactic task
    if ':' not in line:
        list_words = line.split()
        marker = True
        # Check if all the words are in W before adding to the test list
        for word in list_words:
            if word not in W:
                marker = False
            elif word not in model.vocab:
                marker = False
        if marker:
            if s < 6:
                test_semantic.append(line)
            else:
                test_syntactic.append(line)
    else:
        s = s + 1
    line = word_test.readline().strip()

# Not able to test semantic, as there are tests that aren't in the model or in W
print(test_semantic)
print(test_syntactic)

def analogy_w2v(test):
    l = test.split()
    # need to be between 0 and 1
    min_cos_dist = 100
    best_word = ""
    for w in W:
        if w in model.vocab and w not in l:
            vec = model[l[1]] - model[l[0]] + model[l[2]]
            dist = scipy.spatial.distance.cosine(vec, model[w])
            #dist = 1 - sklearn.metrics.pairwise.cosine_similarity(vec.reshape(1,-1), model[w].reshape(1,-1))[0][0]
            if dist < min_cos_dist:
                min_cos_dist = dist
                best_word = w
    return best_word



########################
#### FROM LAST LAB #####
########################

n_W = len(W)
br_w = brown.words()
all_bgrm = bigrams(br_w)
n_all_words = len(br_w)
#mybigrams = dict(ConditionalFreqDist(bigrams(brown.words()))

# Word-context vector model by collecting bigram counts for words in W
M1 = scipy.sparse.lil_matrix((n_W, n_W))
print(M1)
#l_all_bgrm = list(all_bgrm)

sum = 0
for (wx, wy) in all_bgrm:
    if  wx.lower() in W and wy.lower() in W:
        row = W.index(wx.lower())
        col = W.index(wy.lower())
        M1[row, col] += 1
        sum += 1

# Initialize the M1+ matrix
M1plus = scipy.sparse.lil_matrix((n_W, n_W))
# I(w,c) = log2 [ P(w,c) / (P(w)P(c)) ]
# P(w,c): joint probability (co-occurrence) - M1 proportion: M1/sum
# P(w), P(c): marginal probabilities - frequency/number of words
# max( I(w,c) , 0 )
i_all, j_all = M1.nonzero()
for i, j in zip(i_all, j_all):
    if M1[i,j] != 0:
        p_wc = M1[i,j] / sum
        p_w = words[W[i]] / n_all_words
        p_c = words[W[j]] / n_all_words
        # I(w,c) = log2 [ P(w,c) / (P(w)P(c)) ]
        if p_w != 0 and p_c != 0:
            M1plus[i,j] = max(0, math.log(p_wc / (p_w * p_c)))
        else:
            M1plus[i,j] = 0

# latent semantic model 
M1plus_a = M1plus.toarray()
# Principal component analysis (PCA). - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit_transform
# Fit the model with X and apply the dimensionality reduction on X.
# components number -> truncated dimentions
#M2_10 = PCA(n_components = 10).fit_transform(M1plus_a)
#M2_100 = PCA(n_components = 100).fit_transform(M1plus_a)
M2_300 = PCA(n_components = 300).fit_transform(M1plus_a)

########################
########################

def analogy_m2(test):
    l = test.split()
    # need to be between 0 and 1
    min_cos_dist = 1000
    best_word = ""
    for w in W:
        if w not in l:
            vec = M2_300[W.index(l[1])] - M2_300[W.index(l[0])] + M2_300[W.index(l[2])]
            dist = scipy.spatial.distance.cosine(vec, M2_300[W.index(w)])
            #dist = 1 - sklearn.metrics.pairwise.cosine_similarity(vec.reshape(1,-1), M2_300[W.index(w)].reshape(1,-1))[0][0]
            if dist < min_cos_dist and dist > 0:
                min_cos_dist = dist
                best_word = w
    return best_word

accuracy_counter_sem_w2v = 0
accuracy_counter_sem_m2 = 0
for test in test_semantic:
    last_word = test.split()[3]
    print(test)
    w2v_w = analogy_w2v(test)
    m2_w = analogy_m2(test)
    print("Results w2v, m2:", w2v_w, m2_w)
    if w2v_w == last_word:
        accuracy_counter_sem_w2v += 1
    if m2_w == last_word:
        accuracy_counter_sem_m2 += 1

accuracy_counter_syn_w2v = 0
accuracy_counter_syn_m2 = 0
for test in test_syntactic:
    last_word = test.split()[3]
    print(test)
    w2v_w = analogy_w2v(test)
    m2_w = analogy_m2(test)
    print("Results w2v, m2:", w2v_w, m2_w)
    if w2v_w == last_word:
        accuracy_counter_syn_w2v =  accuracy_counter_syn_w2v + 1
    if m2_w == last_word:
        accuracy_counter_syn_m2 = accuracy_counter_syn_m2 + 1
    
print("Accuracy for word2vec semantic:", accuracy_counter_sem_w2v)
print("Accuracy for M2_300 semantic:", accuracy_counter_sem_m2)

print("No of tests:", len(test_semantic))

print("Accuracy for word2vec syntactic:", accuracy_counter_syn_w2v)
print("Accuracy for M2_300 syntactic:", accuracy_counter_syn_m2)

print("No of tests:", len(test_syntactic))

'''
Step 5
'''

'''
One way of improving the exisit set of vector-based models in capturing word
similarities would be also by measuring how common they are in different places.
For example, there exist different analogies in different english-speaking countries
that aren't used in other places as often. However, based on the context, access to
those analogies might be useful. For instance, having an English user saying the 
word lift referring to an elevator, may not reflect is meaning if the model is
mostly american, as the connection to elevator is not frequent.

Another way could be by considering words that have the same general meaning without
looking at a particular form to differentiate for other words who might have other meanings
could originate a more accurate approximation. For instance, we see the test: "work works play plays"
Work and works could be tied to a job and the action to perform a task, but they have
generally the same concept. Whereas play and plays could refer to two totally different
things like the action to play but also performance (dramatic play). This is a more
complicated approach as there would need to be a mix of different processings involved
to be able to differentiate the meaning and group together all the instances where the 
words mean the same.
'''