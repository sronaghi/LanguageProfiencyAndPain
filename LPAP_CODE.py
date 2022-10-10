#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import numpy as np
df = pd.read_csv('Downloads/pain1.csv')


# In[139]:


less = []
inter = []
adv = []
more = []
sample = []
for a,b in zip(df.iloc[:, 3], df.iloc[:, 6]):
    b_str = b[0: (len(b) - 1)]
    new_b = int(b_str)
    if (a == "Novice"):
        less.append(new_b)
    if (a == "Intermediate"):
        inter.append(new_b)
    if (a == "Advanced"):
        adv.append(new_b)
    if (a == "Completely fluent"):
        more.append(new_b)
    sample.append(new_b)
less_mean = np.mean(less)
print(less_mean)
inter_mean = np.mean(inter)
print(inter_mean)
adv_mean = np.mean(adv)
print(adv_mean)
more_mean = np.mean(more)
print(more_mean)
lm_diff = abs(more_mean - less_mean)
im_diff = abs(more_mean - inter_mean)
adm_diff = abs(more_mean - adv_mean)
print(lm_diff)
print(im_diff)
print(adm_diff)
count1 = 0
for i in range(1000000):
    sample1 = np.random.choice(sample, 18, replace = True)
    sample2 = np.random.choice(sample, 18, replace = True)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    mean_diff = abs(mean1 - mean2)
    if (mean_diff > lm_diff):
        count1 += 1 
print(count1 / 1000000)
count4 = 0
for i in range(1000000):
    sample1 = np.random.choice(sample, 18, replace = True)
    sample2 = np.random.choice(sample, 18, replace = True)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    mean_diff = abs(mean1 - mean2)
    if (mean_diff > im_diff):
        count4 += 1 
print(count4 / 1000000)
count5 = 0
for i in range(1000000):
    sample1 = np.random.choice(sample, 18, replace = True)
    sample2 = np.random.choice(sample, 18, replace = True)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    mean_diff = abs(mean1 - mean2)
    if (mean_diff > adm_diff):
        count5 += 1 
print(count5 / 1000000)
count6 = 0
for i in range(1000000):
    samplenov = np.random.choice(sample, 18, replace = True)
    sampleint = np.random.choice(sample, 18, replace = True)
    sampleadv = np.random.choice(sample, 18, replace = True)
    sampleflu = np.random.choice(sample, 18, replace = True)
    mean1 = np.mean(samplenov)
    mean2 = np.mean(sampleint)
    mean3 = np.mean(sampleadv)
    mean4 = np.mean(sampleflu)
    if (mean1 > mean2 and mean2 < mean3 and mean3 > mean4):
        count6 += 1 
print(count6 / 1000000)

count7 = 0
for i in range(1000000):
    samplenov = np.random.choice(sample, 18, replace = True)
    sampleint = np.random.choice(sample, 18, replace = True)
    sampleadv = np.random.choice(sample, 18, replace = True)
    sampleflu = np.random.choice(sample, 18, replace = True)
    mean1 = np.mean(samplenov)
    mean2 = np.mean(sampleint)
    mean3 = np.mean(sampleadv)
    mean4 = np.mean(sampleflu)
    if (mean1 > mean2 and mean2 > mean3 and mean3 > mean4):
        count7 += 1 
print(count7 / 1000000)


# In[49]:


import pandas as pd
import numpy as np
df1 = pd.read_csv('Downloads/spanishdata2.csv')
df1


# In[51]:


def count_words(str):
    word_list = str.split()
    return len(word_list)


# In[141]:


nov = []
inte = []
adva = []
flu = []
nov_len = []
inte_len = []
adva_len = []
flu_len = []
total_words = []
nov_words = []
inte_words = []
adva_words = []
flu_words = []
for a,b in zip(df1.iloc[:, 2], df1.iloc[:, 3]):
    word_list = b.split()
    for i in range (len(word_list)):
        total_words.append(word_list[i])
    print(word_list)
    total_words.append(b.split());
       word_list = b.split
    for word in word_list:
        total_words.append(word)
    if (a == "Novice"):
        nov.append(b)   
        nov_len.append(count_words(b))
        for i in range (len(word_list)):
            nov_words.append(word_list[i])
    if (a == "Intermediate"):
        inte.append(b) 
        inte_len.append(count_words(b))
        for i in range (len(word_list)):
            inte_words.append(word_list[i])
    if (a == "Advanced"):
        adva.append(b) 
        adva_len.append(count_words(b))
        for i in range (len(word_list)):
            adva_words.append(word_list[i])
    if (a == "Completely fluent"):
        flu.append(b) 
        flu_len.append(count_words(b))
        for i in range (len(word_list)):
            flu_words.append(word_list[i])
print(total_words)
plt.hist(total_words)
import matplotlib.pyplot as plt
from collections import Counter
hist = Counter(total_words)
print(hist)
plt.bar(hist.keys() hist.values(), width=5)
hist1 = Counter(nov_words)
print(hist1)
plt.bar(hist1.keys(), hist1.values(), width=5)
hist2 = Counter(adva_words)
print(hist2)
plt.bar(hist2.keys(), hist2.values(), width=5)
hist3 = Counter(inte_words)
print(hist3)
plt.bar(hist3.keys(), hist3.values(), width=5)
hist4 = Counter(flu_words)
print(hist4)
plt.bar(hist4.keys(), hist4.values(), width=5)
# probabilities of the most common words for the different groups 
# dolor 

prob1 = hist1['dolor']/ len(hist1)
prob2 = hist2['dolor']/ len(hist2)
prob3 = hist3['dolor']/ len(hist3)
prob4 = hist4['dolor']/ len(hist4)

print(prob1)
print(prob2)
print(prob3)
print(prob4)

prob9 = hist1['muy']/ len(hist1)
prob10 = hist2['muy']/ len(hist2)
prob11 = hist3['muy']/ len(hist3)
prob12 = hist4['muy']/ len(hist4)

print(prob9)
print(prob10)
print(prob11)
print(prob12)

prob13 = hist1['duele']/ len(hist1)
prob14 = hist2['duele']/ len(hist2)
prob15 = hist3['duele']/ len(hist3)
prob16 = hist4['duele']/ len(hist4)

print(prob13)
print(prob14)
print(prob15)
print(prob16)

nov_mean = np.mean(nov_len)
inte_mean = np.mean(inte_len)
adva_mean = np.mean(adva_len)
flu_mean = np.mean(flu_len)
print(nov_mean)
print(inte_mean)
print(adva_mean)
print(flu_mean)
nf_diff = abs(flu_mean - nov_mean)
if_diff = abs(flu_mean - inte_mean)
af_diff = abs(flu_mean - adva_mean)
print(nf_diff)
print(if_diff)
print(af_diff)
count1 = 0
for i in range(1000000):
    sample1 = np.random.choice(sample, 6, replace = True)
    sample2 = np.random.choice(sample, 6, replace = True)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    mean_diff = abs(mean1 - mean2)
    if (mean_diff > nf_diff):
        count1 += 1 
print(count1 / 1000000)
count2 = 0
for i in range(1000000):
    sample1 = np.random.choice(sample, 6, replace = True)
    sample2 = np.random.choice(sample, 6, replace = True)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    mean_diff = abs(mean1 - mean2)
    if (mean_diff > if_diff):
        count2 += 1 
print(count2 / 1000000)
count3 = 0
for i in range(1000000):
    sample1 = np.random.choice(sample, 6, replace = True)
    sample2 = np.random.choice(sample,6, replace = True)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    mean_diff = abs(mean1 - mean2)
    if (mean_diff > af_diff):
        count3 += 1 
print(count3 / 1000000)

