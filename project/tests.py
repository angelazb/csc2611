# Before "('cours', 'work')",44
# After "('cours', 'work')",101
# Before "('class', 'work')",15
# After "('class', 'work')",29

# Before "('procrastin', 'time')",18
# After "('procrastin', 'time')",4

from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chisquare
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import statsmodels.stats.weightstats
# defining the table 
#data = [[207, 282, 241], [234, 242, 232]] 
# First column is course work and second is class work
#data = [[44, 15, 18], [101, 29, 4]]


N = 10000

filename = "files/Both_Prepare_Start_Words.csv"
df_main = pd.read_csv(filename, encoding="ISO-8859-1")

filename = "files/Winter_Prepare_Start_Words.csv"
df_winter = pd.read_csv(filename, encoding="ISO-8859-1")

filename = "files/Fall_Prepare_Start_Words.csv"
df_fall = pd.read_csv(filename, encoding="ISO-8859-1")

main = []
winter = []
fall = []
main_c = []
new_main = []
for i in range(500):
    bigram = df_main['prepare_start_bigram'][i]
    new_main.append(df_main['normalized_count'][i]*N)
    main.append(bigram)
    w_l = list(df_winter['prepare_start_bigram'])
    f_l = list(df_fall['prepare_start_bigram'])

    if bigram in w_l:
        w_i = w_l.index(bigram)
        winter.append(df_winter['normalized_count'][w_i]*N)
    else:
        winter.append(0)
    if bigram in f_l:
        f_i = f_l.index(bigram)
        fall.append(df_fall['normalized_count'][f_i]*N)
    else:
        fall.append(0)

result = pd.DataFrame({'bigram' : main})
result["before"] = winter
result["after"] = fall
result["total"] = new_main
#winter_p = []
#fall_p = []
#for i in range(len(winter)):
#    main_c.append(int(winter[i]) + int(fall[i]))
#    winter_p.append((int(winter[i])/sum(winter))*1000)
#    fall_p.append((int(fall[i])/sum(fall))*1000)
#result["total"] = main_c
#print(winter)
#print(fall)

exp = []

#for i in range(40):
#    val = ((fall[i]-winter[i])**2)/winter[i]
#    exp.append(val)
data = [winter,fall]

tStat, pValue = stats.ttest_ind(winter, fall, equal_var = False) #run independent sample T-Test
print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic

sns.kdeplot(winter, shade=True, label="Winter")
sns.kdeplot(fall, shade=True, label="Fall")
plt.title("Winter vs Fall Prepare Words Frequency Independent T-Test")
plt.legend()
plt.xlabel("P-Value:{0} T-Statistic:{1}".format(pValue,tStat))
plt.show()

ztest ,pval1 = statsmodels.stats.weightstats.ztest(winter, x2=fall, value=0,alternative='two-sided')
print(float(pval1))
result.to_csv("files/CT_Prepare_N_500.csv", index=False)
data=[winter, fall]
stat, p, dof, expected = chi2_contingency(data)

#for i in range(len(expected[0]))
# interpret p-value 
alpha = 0.05
print("p value is " + str(p)) 
print("stat is "+ str(stat) + " --- dof is " + str(dof))
#print(" --- expected is " + str(expected))
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 

#stat, p = chisquare(np.array(winter), np.array(new_main), axis=None)
#
#print("p value is " + str(p)) 
#print("stat is "+ str(stat))

#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('Chi-Square Distribution')
#plt.show()

