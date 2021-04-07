import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "files/CT_Perform.csv"
df = pd.read_csv(filename, encoding="ISO-8859-1")

words = list(df['bigram'][:20])
w_l = list(df['before'][:20])
f_l = list(df['after'][:20])


x = np.arange(len(words))
width = 0.25

fig, ax = plt.subplots()
winter = ax.bar(x - width/2, w_l, width, label="Winter")
fall = ax.bar(x + width/2, f_l, width, label="Fall")

ax.set_title("Winter vs Fall Perform Top 20 Word Frequency")
ax.set_ylabel("Count")
ax.set_xlabel("Top 20 Words")
ax.set_xticks(x)
ax.set_xticklabels(words, rotation=90)
ax.legend()

fig.tight_layout()
plt.show()