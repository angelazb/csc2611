import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

filename = "files/Neut_Winter_Prepare_Sentiment.csv"
w_prep_df = pd.read_csv(filename, encoding="ISO-8859-1")
w_prep_sent = Counter(list(w_prep_df['sent_word']))
pos = w_prep_sent['pos']
neg = w_prep_sent['neg']
neut = w_prep_sent['neutral']
total = pos + neg + neut
print("Winter Prepare")
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos, neg, neut))
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos/total, neg/total, neut/total))

filename = "files/Neut_Fall_Prepare_Sentiment.csv"
f_prep_df = pd.read_csv(filename, encoding="ISO-8859-1")
f_prep_sent = Counter(list(f_prep_df['sent_word']))
pos = f_prep_sent['pos']
neg = f_prep_sent['neg']
neut = f_prep_sent['neutral']
total = pos + neg + neut
print("Fall Prepare")
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos, neg, neut))
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos/total, neg/total, neut/total))

filename = "files/Neut_Winter_Perform_Sentiment.csv"
w_perf_df = pd.read_csv(filename, encoding="ISO-8859-1")
w_perf_sent = Counter(list(w_perf_df['sent_word']))
pos = w_perf_sent['pos']
neg = w_perf_sent['neg']
neut = w_perf_sent['neutral']
total = pos + neg + neut
print("Winter Perform")
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos, neg, neut))
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos/total, neg/total, neut/total))

filename = "files/Neut_Fall_Perform_Sentiment.csv"
f_perf_df = pd.read_csv(filename, encoding="ISO-8859-1")
f_perf_sent = Counter(list(f_perf_df['sent_word']))
pos = f_perf_sent['pos']
neg = f_perf_sent['neg']
neut = f_perf_sent['neutral']
total = pos + neg + neut
print("Fall Perform")
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos, neg, neut))
print("Positive={0} - Negative={1} - Neutral={2}\n".format(pos/total, neg/total, neut/total))

