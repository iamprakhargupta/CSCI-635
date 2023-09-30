#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Using pandas to read csv
"""
file: q1.py
description: Ploting and stats on the 2 frogs datasets

language: python3
author: Prakhar Gupta pg9349

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams


# In[2]:


df_sample=pd.read_csv("Frogs-subsample.csv")


# In[3]:


df_sample.head()


# In[ ]:





# In[4]:


df_sample.shape


# In[5]:


df=pd.read_csv("Frogs.csv")


# In[6]:


print("shape",df.shape)
print("----------------------------------------------------")


# In[7]:


print(df['Species'].value_counts())
print("----------------------------------------------------")


# In[58]:


### For main data
## Scatter plot for main data
# figure size in inches
print("Plotting Raw Features")
print("----------------------------------------------------")
rcParams['figure.figsize'] = 11.7,8.27
print("Scatter plot of complete data")
sns.scatterplot(data=df, x="MFCCs_10", y="MFCCs_17",hue='Species')
plt.title("Scatter plot of complete data")
plt.show()
print("----------------------------------------------------")


# In[59]:


## Scatter plot for sample data
rcParams['figure.figsize'] = 11.7,8.27
print("Scatter plot of sample data")
sns.scatterplot(data=df_sample, x="MFCCs_10", y="MFCCs_17",hue='Species')
plt.title("Scatter plot of sample data")
plt.show()
print("----------------------------------------------------")


# In[10]:


#### Separating into classes.
df_sample_class1=df_sample[df_sample.Species=="HylaMinuta"]
df_sample_class2=df_sample[df_sample.Species=="HypsiboasCinerascens"]


# In[11]:


df_class1=df[df.Species=="HylaMinuta"]
df_class2=df[df.Species=="HypsiboasCinerascens"]


# In[61]:


## For sample data 
##1. Histograms
# For  HylaMinuta
plt.hist(df_sample_class1["MFCCs_10"],11)
plt.xlabel('MFCCs_10')
plt.ylabel('Counts')
print('Sample distribution For HylaMinuta MFCC 10')
plt.title('Sample distribution For HylaMinuta MFCC 10')
plt.show()
print("----------------------------------------------------")


# In[62]:


# For MFCC 17
print('Sample distribution For HylaMinuta MFCC 17')
plt.hist(df_sample_class1["MFCCs_17"],11)
plt.xlabel('MFCCs_17')
plt.ylabel('Counts')
plt.title('Sample distribution For HylaMinuta MFCC 17')
plt.show()
print("----------------------------------------------------")


# In[64]:


# For  HypsiboasCinerascens
print("Sample distribution For HypsiboasCinerascens MFCC 10")
plt.hist(df_sample_class2["MFCCs_10"],11)
plt.xlabel('MFCCs_10')
plt.ylabel('Counts')
plt.title('Sample distribution For HypsiboasCinerascens MFCC 10')
plt.show()
print("----------------------------------------------------")


# In[65]:


print("Sample distribution For HypsiboasCinerascens MFCC 17")
plt.hist(df_sample_class2["MFCCs_17"],11)
plt.xlabel('MFCCs_17')
plt.ylabel('Counts')
plt.title('Sample distribution For HypsiboasCinerascens MFCC 17')
plt.show()

print("----------------------------------------------------")


# In[66]:


## For Complete data data 
##1. Histograms
# For  HylaMinuta
print("Complete data distribution For HylaMinuta MFCC 10")
plt.hist(df_class1["MFCCs_10"],20)
plt.xlabel('MFCCs_10')
plt.ylabel('Counts')
plt.title('Complete data distribution For HylaMinuta MFCC 10')
plt.show()
print("----------------------------------------------------")


# In[67]:


print('Complete data distribution For HylaMinuta MFCC 17')
plt.hist(df_class1["MFCCs_17"],20)
plt.xlabel('MFCCs_17')
plt.ylabel('Counts')
plt.title('Complete data distribution For HylaMinuta MFCC 17')
plt.show()
print("----------------------------------------------------")


# In[68]:


# For  HypsiboasCinerascens
print('Complete data distribution For HypsiboasCinerascens MFCC 10')
plt.hist(df_class2["MFCCs_10"],20)
plt.xlabel('MFCCs_10')
plt.ylabel('Counts')
plt.title('Complete data distribution For HypsiboasCinerascens MFCC 10')
plt.show()
print("----------------------------------------------------")


# In[69]:


print('Complete data distribution For HypsiboasCinerascens MFCC 17')
plt.hist(df_class2["MFCCs_17"],20)
plt.xlabel('MFCCs_17')
plt.ylabel('Counts')
plt.title('Complete data distribution For HypsiboasCinerascens MFCC 17')
plt.show()
print("----------------------------------------------------")


# In[20]:


### Line plot


# In[21]:


### Sample Data Line plots


# In[71]:


class1=df_sample_class1.sort_values("MFCCs_10")


# In[72]:


print('Lineplot for MFCC 10 For HylaMinuta (Sample data)')
plt.plot(sorted(class1.index),class1["MFCCs_10"])
plt.xlabel('index')
plt.ylabel('MFCCs_10')
plt.title('Lineplot for MFCC 10 For HylaMinuta (Sample data)')
plt.show()
print("----------------------------------------------------")


# In[74]:


class1=df_sample_class1.sort_values("MFCCs_17")


# In[75]:


print("Lineplot for MFCC 17 For HylaMinuta  (Sample data)")
plt.plot(sorted(class1.index),class1["MFCCs_17"])
plt.xlabel('index')
plt.ylabel('MFCCs_17')
plt.title('Lineplot for MFCC 17 For HylaMinuta  (Sample data)')
plt.show()
print("----------------------------------------------------")


# In[76]:


class2=df_sample_class2.sort_values("MFCCs_10")


# In[77]:


print('Lineplot for MFCC 10 For HypsiboasCinerascens  (Sample data)')
plt.plot(sorted(class2.index),class2["MFCCs_10"])
plt.xlabel('index')
plt.ylabel('MFCCs_10')
plt.title('Lineplot for MFCC 10 For HypsiboasCinerascens  (Sample data)')
plt.show()
print("----------------------------------------------------")


# In[79]:


class2=df_sample_class2.sort_values("MFCCs_17")


# In[80]:


print('Lineplot for MFCC 17 For HypsiboasCinerascens  (Sample data)')
plt.plot(sorted(class2.index),class2["MFCCs_17"])
plt.xlabel('index')
plt.ylabel('MFCCs_17')
plt.title('Lineplot for MFCC 17 For HypsiboasCinerascens  (Sample data)')
plt.show()

print("----------------------------------------------------")


# In[30]:


### Complete Data Line plots
print("FOr complete data")
print("----------------------------------------------------")


# In[82]:


class1=df_class1.sort_values("MFCCs_10")


# In[83]:


print('Lineplot for MFCC 10 For HylaMinuta (Complete Data)')
plt.plot(sorted(class1.index),class1["MFCCs_10"])
plt.xlabel('index')
plt.ylabel('MFCCs_10')
plt.title('Lineplot for MFCC 10 For HylaMinuta (Complete Data)')
plt.show()

print("----------------------------------------------------")


# In[84]:


class1=df_class1.sort_values("MFCCs_17")


# In[85]:


print('Lineplot for MFCC 17 For HylaMinuta (Complete Data)')
plt.plot(sorted(class1.index),class1["MFCCs_17"])
plt.xlabel('index')
plt.ylabel('MFCCs_17')
plt.title('Lineplot for MFCC 17 For HylaMinuta (Complete Data)')
plt.show()

print("----------------------------------------------------")


# In[87]:


class2=df_class2.sort_values("MFCCs_10")


# In[88]:


print('Lineplot for MFCC 10 For  HypsiboasCinerascens (Complete Data)')
plt.plot(sorted(class2.index),class2["MFCCs_10"])
plt.xlabel('index')
plt.ylabel('MFCCs_10')
plt.title('Lineplot for MFCC 10 For  HypsiboasCinerascens (Complete Data)')
plt.show()

print("----------------------------------------------------")


# In[90]:


class2=df_class2.sort_values("MFCCs_17")


# In[91]:


print('Lineplot for MFCC 17 For  HypsiboasCinerascens (Complete Data)')
plt.plot(sorted(class2.index),class2["MFCCs_17"])
plt.xlabel('index')
plt.ylabel('MFCCs_17')
plt.title('Lineplot for MFCC 17 For  HypsiboasCinerascens (Complete Data)')
plt.show()

print("----------------------------------------------------")


# In[39]:


print("Plotting Feature Distributions")
#Plotting Feature Distributions


# In[94]:


# Box Plots
print("BoxPlots")
print("For sample data")
df_sample.boxplot(by="Species",)

plt.show()
print("----------------------------------------------------")


# In[95]:


print("BoxPlots")
print("For complete data")
df.boxplot(by="Species")
plt.show()
print("----------------------------------------------------")


# In[96]:


print("Bar chart for sample data")
df_sample_melt=pd.melt(df_sample, id_vars=['Species'] ,value_vars=['MFCCs_10',"MFCCs_17"])
rcParams['figure.figsize'] = 15,10
sns.barplot(x='variable',y='value',hue="Species",data=df_sample_melt)
plt.title("Bar chart for sample data")
plt.plot()
print("----------------------------------------------------")


# In[97]:


print("Bar chart for complete data")
df_melt=pd.melt(df, id_vars=['Species'] ,value_vars=['MFCCs_10',"MFCCs_17"])
rcParams['figure.figsize'] = 15,10
sns.barplot(x='variable',y='value',hue="Species",data=df_melt)
plt.title("Bar chart for complete data")
plt.plot()
print("----------------------------------------------------")


# In[44]:


#Descriptive Statistics


# In[45]:


## For sample data


# In[46]:


import numpy as np


# In[47]:


print("For sample data")


# In[48]:



print("Mean for HylaMinuta")
print(np.mean(df_sample_class1))
print("----------------------------------------------------")


# In[49]:


print("Mean for HypsiboasCinerascens")
print(np.mean(df_sample_class2))
print("----------------------------------------------------")


# In[50]:


print("Standard deviation for HylaMinuta")
print(np.std(df_sample_class1))
print("----------------------------------------------------")


# In[51]:


print("Standard deviations for HypsiboasCinerascens")
print(np.std(df_sample_class2))
print("----------------------------------------------------")


# In[52]:


print("Covariance matrix for HylaMinuta")
l=['MFCCs_10','MFCCs_17']
j=0
print("        ",l)
for i in np.cov(df_sample_class1['MFCCs_10'],df_sample_class1['MFCCs_17']):
    print(l[j],i)
    j+=1
print("----------------------------------------------------")    


# In[53]:


print("Covariance matrix for HypsiboasCinerascens")
l=['MFCCs_10','MFCCs_17']
j=0
print("        ",l)
for i in np.cov(df_sample_class2['MFCCs_10'],df_sample_class2['MFCCs_17']):
    print(l[j],i)
    j+=1
print("----------------------------------------------------")    


# In[54]:


print("For Complete data")


# In[56]:



print("Mean for HylaMinuta")
print(np.mean(df_class1))
print("----------------------------------------------------")
print("Mean for HypsiboasCinerascens")
print(np.mean(df_class2))
print("----------------------------------------------------")
print("Standard deviation for HylaMinuta")
print(np.std(df_class1))
print("----------------------------------------------------")
print("Standard deviations for HypsiboasCinerascens")
print(np.std(df_class2))
print("----------------------------------------------------")
print("Covariance matrix for HylaMinuta")
l=['MFCCs_10','MFCCs_17']
j=0
print("        ",l)
for i in np.cov(df_class1['MFCCs_10'],df_class1['MFCCs_17']):
    print(l[j],i)
    j+=1
print("----------------------------------------------------")    

print("Covariance matrix for HypsiboasCinerascens")
l=['MFCCs_10','MFCCs_17']
j=0
print("        ",l)
for i in np.cov(df_class2['MFCCs_10'],df_class2['MFCCs_17']):
    print(l[j],i)
    j+=1    
print("----------------------------------------------------")    


# In[ ]:




