import numpy as np
import pickle
import sklearn.metrics
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

'''
Step 1
'''
d = open("embeddings/data.pkl", "rb")
diac = pickle.load(d)
#print(diac)

#'w': a list of 2000 words, a subset of the English lexicon
W = diac["w"]
#'d': a list of decades between 1900 and 2000
Y = diac["d"]
#'E': a 2000 by 10 by 300 list of list of vectors; the (i,j)-th entry is a 300-dimensional vector for the i-th word in the j-th decade
E = diac["E"]

'''
Step 2
'''
### Cosine similarity between the first and last decade
# cosine distance = 1 - cosine similarity
cosine_similarities = []
cosine_distance = []
def cosine_sim():
    for i in range(len(W)):
        # sklearn.metrics.pairwise.cosine_similarity(X, Y=None, dense_output=True)
        # https://www.programcreek.com/python/example/100424/sklearn.metrics.pairwise.cosine_similarity - Example 3
        row1 = E[i][0]
        row2 = E[i][-1]
        cosine_similarities.append(sklearn.metrics.pairwise.cosine_similarity(row1.reshape(1,-1), row2.reshape(1,-1))[0][0])
        cosine_distance.append(1 - cosine_similarities[i])

### Max cosine distance between each consecutive year for every word
max_cosine_distance = []
def max_cosine():
    for i in range(len(W)):
        max_cos = 0
        for year in range(9):
            row1 = E[i][year]
            row2 = E[i][year + 1]
            cosine_distance = 1 - sklearn.metrics.pairwise.cosine_similarity(row1.reshape(1,-1), row2.reshape(1,-1))[0][0]
            max_cos = max(max_cos, cosine_distance)
        max_cosine_distance.append(max_cos)

### Average of distances between each consecutive year for every word
avg_cosine_distance = []
def avg_cosine():
    for i in range(len(W)):
        cos_dist = 0
        for year in range(9):
            row1 = E[i][year]
            row2 = E[i][year + 1]
            cosine_distance = 1 - sklearn.metrics.pairwise.cosine_similarity(row1.reshape(1,-1), row2.reshape(1,-1))[0][0]
            cos_dist = cos_dist + cosine_distance
        avg_cosine_distance.append(cos_dist / 9)

cosine_sim()
max_cosine()
avg_cosine()

print(cosine_distance)
print(max_cosine_distance)
print(avg_cosine_distance)

def find_largests(l):
    copy_list = list(enumerate(l))
    words = []
    values = []

    for i in range(20):
        max_n = -1000
        max_i = -1000
        for j in range(len(copy_list)):
            if copy_list[j][1] > max_n:
                max_n = copy_list[j][1]
                max_i = copy_list[j][0]
        copy_list.remove((max_i, max_n))
        values.append(max_n)
        words.append(W[max_i])
    return words

def find_smallest(l):
    copy_list = list(enumerate(l))
    words = []
    values = []

    for i in range(20):
        min_n = 1000
        min_i = 1000
        for j in range(len(copy_list)):
            if copy_list[j][1] < min_n:
                min_n = copy_list[j][1]
                min_i = copy_list[j][0]
        copy_list.remove((min_i, min_n))
        values.append(min_n)
        words.append(W[min_i])
    return words

cosine_distance_most_c = find_largests(cosine_distance)
cosine_distance_least_c = find_smallest(cosine_distance)
max_cosine_most_c = find_largests(max_cosine_distance)
max_cosine_least_c = find_smallest(avg_cosine_distance)
avg_cosine_most_c = find_largests(max_cosine_distance)
avg_cosine_least_c = find_smallest(avg_cosine_distance)

print("Top 20: Cosine distance first to last decade", cosine_distance_most_c)
print("Least 20: Cosine distance first to last decade", cosine_distance_least_c)
print("Top 20: Max cosine distance", max_cosine_most_c)
print("Least 20: Max cosine distance", max_cosine_least_c)
print("Top 20: Avg cosine distance", avg_cosine_most_c)
print("Least 20: Avg cosine distance", avg_cosine_least_c)

print("Pearson corr: First-Last and Max", pearsonr(cosine_distance, max_cosine_distance))
print("Pearson corr: First-Last and Avg", pearsonr(cosine_distance, avg_cosine_distance))
print("Pearson corr: Max and Avg", pearsonr(max_cosine_distance, avg_cosine_distance))

'''
Step 3
'''

# Check the closest 5 words to each other word on the first and last decade

def closest(word, year):
    distances = []
    for i in range(2000):
        row1 = E[W.index(word)][year]
        row2 = E[i][year]
        cosine_distance = 1 - sklearn.metrics.pairwise.cosine_similarity(row1.reshape(1,-1), row2.reshape(1,-1))[0][0]
        distances.append((W[i], cosine_distance))
    d = sorted(distances, key=lambda x: (-x[1],x[0]))
    return [t[0] for t in d][:5]

comparison = []
def closest_acc():
    for i in range(2000):
        first = set(closest(W[i], 0))
        last = set(closest(W[i], 9))
        comparison.append(1-len(first.intersection(last))/10)

closest_acc()
print("Pearson corr: First-Last and Evaluation", pearsonr(comparison,cosine_distance))
print("Pearson corr: Max and Evaluation", pearsonr(comparison, max_cosine_distance))
print("Pearson corr: Avg and Evaluation", pearsonr(comparison, avg_cosine_distance))

'''
Step 5
'''
def cos_per_year(w, i):
    cosines_per_year = []
    for year in range(9):
        row1 = E[i][year]
        row2 = E[i][year + 1]
        cosines_per_year.append(1 - sklearn.metrics.pairwise.cosine_similarity(row1.reshape(1,-1), row2.reshape(1,-1))[0][0])
    return cosines_per_year

w1 = "techniques"
i1 = W.index(w1)
c1 = cos_per_year(w1, i1)
print(c1)

w2 = "computer"
i2 = W.index(w2)
c2 = cos_per_year(w2, i2)
print(c2)

w3 = "skills"
i3 = W.index(w3)
c3 = cos_per_year(w3, i3)
print(c3)

plt.scatter(Y[1:], c1)
plt.plot(Y[1:], c1, label='techinques')
plt.scatter(Y[1:], c2)
plt.plot(Y[1:], c2, label="computer")
plt.scatter(Y[1:], c3)
plt.plot(Y[1:], c3, label="skills")
plt.xlabel('Time')
plt.ylabel('Cosine Distance')
plt.legend()
plt.title('Semantic change of techniques, computer, and skills')
#plt.show()