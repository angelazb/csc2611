import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

filename = "files/Winter_Prepare_Sentiment.csv"
w_prep_df = pd.read_csv(filename, encoding="ISO-8859-1")
w_prep_sent = Counter(list(w_prep_df['sent_word']))
pos = w_prep_sent['pos']
neg = w_prep_sent['neg']
total = pos + neg
print("Winter Prepare")
print("Positive={0} - Negative={1}\n".format(pos, neg))
print("Positive={0} - Negative={1}\n".format(pos/total, neg/total))

filename = "files/Fall_Prepare_Sentiment.csv"
f_prep_df = pd.read_csv(filename, encoding="ISO-8859-1")
f_prep_sent = Counter(list(f_prep_df['sent_word']))
pos = f_prep_sent['pos']
neg = f_prep_sent['neg']
total = pos + neg
print("Fall Prepare")
print("Positive={0} - Negative={1}\n".format(pos, neg))
print("Positive={0} - Negative={1}\n".format(pos/total, neg/total))

filename = "files/Winter_Perform_Sentiment.csv"
w_perf_df = pd.read_csv(filename, encoding="ISO-8859-1")
w_perf_sent = Counter(list(w_perf_df['sent_word']))
pos = w_perf_sent['pos']
neg = w_perf_sent['neg']
total = pos + neg
print("Winter Perform")
print("Positive={0} - Negative={1}\n".format(pos, neg))
print("Positive={0} - Negative={1}\n".format(pos/total, neg/total))

filename = "files/Fall_Perform_Sentiment.csv"
f_perf_df = pd.read_csv(filename, encoding="ISO-8859-1")
f_perf_sent = Counter(list(f_perf_df['sent_word']))
pos = f_perf_sent['pos']
neg = f_perf_sent['neg']
total = pos + neg
print("Fall Perform")
print("Positive={0} - Negative={1}\n".format(pos, neg))
print("Positive={0} - Negative={1}\n".format(pos/total, neg/total))

