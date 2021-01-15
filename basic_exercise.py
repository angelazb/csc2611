'''
STEP 1
'''
from nltk import *
from nltk.corpus import brown
from nltk.book import *
import numpy as np
import scipy
import math
from sklearn.decomposition import PCA
import sklearn

'''
STEP 2
'''

# All words from the brown corpus
# Extract the 5000 most common words denoted by W
words = FreqDist(brown.words())
W_tuples = words.most_common(5000)
W = [t[0].lower() for t in W_tuples]

# Report the 5 most and least common words found
print("The 5 most common words found", W[:5])
print("The 5 least common words found", W[-5:])

# Words in Table 1 of RG65
RG65 = {"cord", "smile", "rooster", "voyage", "noon", "string", "fruit", "furnance", "autograph", "shore", "automobile", "wizard", "mound", "stove", "grin", "implement", "asylum", "fruit", "asylum", "monk", "graveyard", "madhouse", "glass", "magician", "boy", "rooster", "cushion", "jewel", "monk", "slave", "asylum", "cementery", "coast", "forest", "grin", "lad", "shore", "woodland", "monk", "oracle", "boy", "sage", "automobile", "cushion", "mound", "shore", "lad", "wizard", "forest", "graveyard", "food", "rooster", "cementary", "woodland", "shore", "voyage", "bird", "woodland", "coast", "hill", "furnace", "implement", "crane", "rooster", "hill", "woodland", "car", "journey", "cementary", "mound", "glass", "jewel", "magician", "oracle", "crane", "implement", "brother", "lad", "sage", "wizard", "oracle", "sage", "bird", "crane", "bird", "cock", "food", "fruit", "brother", "monk", "asylum", "madhouse", "furnace", "stove", "magician", "wizard", "hill", "mound", "cord", "string", "glass", "tumbler", "grin", "smile", "serf", "slave", "journey", "voyage", "autograph", "signature", "coast", "shore", "forest", "woodland", "implement", "tool", "cock", "rooster", "boy", "lad", "cushion", "pillow", "cementery", "graveyard", "automobile", "car", "midday", "noon", "gem", "jewel"}
# Word pairs in Table 1 of RG65
RG65_pairs = [("cord", "smile"), ("rooster", "voyage"), ("noon", "string"), ("fruit", "furnance"), ("autograph", "shore"), ("automobile", "wizard"), ("mound", "stove"), ("grin", "implement"), ("asylum", "fruit"), ("asylum", "monk"), ("graveyard", "madhouse"), ("glass", "magician"), ("boy", "rooster"), ("cushion", "jewel"), ("monk", "slave"), ("asylum", "cementery"), ("coast", "forest"), ("grin", "lad"), ("shore", "woodland"), ("monk", "oracle"), ("boy", "sage"), ("automobile", "cushion"), ("mound", "shore"), ("lad", "wizard"), ("forest", "graveyard"), ("food", "rooster"), ("cementary", "woodland"), ("shore", "voyage"), ("bird", "woodland"), ("coast", "hill"), ("furnace", "implement"), ("crane", "rooster"), ("hill", "woodland"), ("car", "journey"), ("cementary", "mound"), ("glass", "jewel"), ("magician", "oracle"), ("crane", "implement"), ("brother", "lad"), ("sage", "wizard"), ("oracle", "sage"), ("bird", "crane"), ("bird", "cock"), ("food", "fruit"), ("brother", "monk"), ("asylum", "madhouse"), ("furnace", "stove"), ("magician", "wizard"), ("hill", "mound"), ("cord", "string"), ("glass", "tumbler"), ("grin", "smile"), ("serf", "slave"), ("journey", "voyage"), ("autograph", "signature"), ("coast", "shore"), ("forest", "woodland"), ("implement", "tool"), ("cock", "rooster"), ("boy", "lad"), ("cushion", "pillow"), ("cementery", "graveyard"), ("automobile", "car"), ("midday", "noon"), ("gem", "jewel")]
RG65_pairs_vals = [0.02, 0.04, 0.04, 0.05, 0.06, 0.11, 0.14, 0.18, 0.19, 0.39, 0.42, 0.44, 0.44, 0.45, 0.57, 0.79, 0.85, 0.88, 0.90, 0.91, 0.96, 0.97, 0.97, 0.99, 1.00, 1.09, 1.18, 1.22, 1.24, 1.26, 1.37, 1.41, 1.48, 1.55, 1.69, 1.78, 1.82, 2.37, 2.41, 2.46, 2.61, 2.63, 2.63, 2.69, 2.74, 3.04, 3.11, 3.21, 3.29, 3.41, 3.45, 3.46, 3.46, 3.58, 3.59, 3.60, 3.65, 3.66, 3.68, 3.82, 3.84, 3.88, 3.92, 3.94, 3.94]
n = len(RG65)
added = 0
# Check how many words aren't yet in W
for word in RG65:
    if word not in W:
        W.append(word)
        added+=1
# |W|
n_W = len(W)
print("From the", str(n), "words found in RG65 Table 1 only", str(added),"were new. Now we have", str(n_W), "words in W")

'''
STEP 3
'''
# 
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

'''
STEP 4
'''
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

'''
STEP 5
'''
# latent semantic model 
M1plus_a = M1plus.toarray()
# Principal component analysis (PCA). - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.fit_transform
# Fit the model with X and apply the dimensionality reduction on X.
# components number -> truncated dimentions
M2_10 = PCA(n_components = 10).fit_transform(M1plus_a)
M2_100 = PCA(n_components = 100).fit_transform(M1plus_a)
M2_300 = PCA(n_components = 300).fit_transform(M1plus_a)

'''
STEP 6
'''
# P = RG65_pairs in all_bgrm, because W is in there, it should be the same.
P = RG65_pairs
S = RG65_pairs_vals

'''
STEP 7
'''
S_M1 = []
S_M2_10 = []
S_M2_100 = []
S_M2_300 = []

def cosine_sim(model, similarities):
    for word1,word2 in P:
        w1 = W.index(word1)
        w2 = W.index(word2)
        
        #model = model.toarray()
        # sklearn.metrics.pairwise.cosine_similarity(X, Y=None, dense_output=True)
        # https://www.programcreek.com/python/example/100424/sklearn.metrics.pairwise.cosine_similarity - Example 3
        row1 = model[w1]
        row2 = model[w2]
        similarities.append(sklearn.metrics.pairwise.cosine_similarity(row1.reshape(1,-1), row2.reshape(1,-1))[0][0])

cosine_sim(M1, S_M1)
cosine_sim(M2_10, S_M2_10)
cosine_sim(M2_100, S_M2_100)
cosine_sim(M2_300, S_M2_300)
'''
STEP 8
'''

print("Pearson correlation between S and the model-predicted similarities:")
print("> M1: ", scipy.stats.pearsonr(S_M1, S))
print("> M2_10: ",scipy.stats.pearsonr(S_M2_10, S))
print("> M2_100: ",scipy.stats.pearsonr(S_M2_100, S))
print("> M2_300: ",scipy.stats.pearsonr(S_M2_300, S))