#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
file: q2.py
description: Create a logistic model using pytorch on the 2 frogs datasets

language: python3
author: Prakhar Gupta pg9349

"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import torch.nn as nn
import numpy as np
### Setting the seed for reproducibity 
torch.manual_seed(0)


# In[2]:


df_sample=pd.read_csv("Frogs-subsample.csv")


# In[3]:


df_sample.columns
df_sample = df_sample.sample(frac=1,random_state=42).reset_index(drop=True)


# In[4]:


trainpercent=0.80
df_sample['Species_encoded']=df_sample.Species.map({"HypsiboasCinerascens":0,"HylaMinuta":1})
train=df_sample.sample(frac=trainpercent,random_state=102) #random state is a seed value
test=df_sample.drop(train.index)


# In[5]:


train.shape,test.shape


# In[6]:


X_train=train[['MFCCs_10', 'MFCCs_17']].to_numpy()

y_train=train[['Species_encoded']].to_numpy()
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))


X_test=test[['MFCCs_10', 'MFCCs_17']].to_numpy()

y_test=test[['Species_encoded']].to_numpy()
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
                           


# In[7]:


df_sample.columns


# In[8]:



class GLm(torch.nn.Module):
    """
    Class to inherit pytorch module
    """
    def __init__(self, inpdim=2, outdim=1):
        super(GLm, self).__init__()
        self.linear = torch.nn.Linear(inpdim, outdim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


# In[9]:


logistic=GLm(2,1)


# In[10]:


learning_rate = 0.01                               
criterion = nn.BCELoss()
passes=2000
optimizer = torch.optim.SGD(logistic.parameters(), lr=learning_rate)


# In[11]:


for epoch in range(passes):
#     print(epoch)
    y_pred = logistic(X_train)
    loss = criterion(y_pred, y_train)             
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 100 == 0:                                         
       
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


# In[12]:


print("Training Accuarcy")
y_predicted = logistic(X_train)
y_predicted_cls = y_predicted.round()
acc = y_predicted_cls.eq(y_train).sum() / float(y_train.shape[0])
print(f'accuracy: {acc.item():.4f}')


# In[13]:


print("Test Accuarcy")
y_predicted = logistic(X_test)
y_predicted_cls = y_predicted.round()
acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
print(f'accuracy: {acc.item():.4f}')


# In[14]:


logistic.linear


# In[15]:


l = logistic.state_dict()
w1,w2=np.array(l['linear.weight'])[0]
w0=np.array(l['linear.bias'])[0]


# In[16]:


c=-(w0/w2)
m=-(w1/w2)


# In[17]:


x1min,x1max=df_sample["MFCCs_10"].min(),df_sample["MFCCs_10"].max()
x2min,x2max=df_sample["MFCCs_17"].min(),df_sample["MFCCs_17"].max()

xd = np.array([x1min, x1max])
yd = m*xd + c


# In[18]:


x1min,x1max


# In[19]:


xd,yd


# In[20]:


rcParams['figure.figsize'] = 11.7,8.27
sns.scatterplot(data=train, x="MFCCs_10", y="MFCCs_17",hue='Species',palette=dict(HypsiboasCinerascens="#298CEE", HylaMinuta="#EE8C29"))
plt.plot(xd, yd, 'k', lw=1.5, ls='--')
plt.fill_between(xd, yd, min(yd), color='tab:blue', alpha=0.1)
plt.fill_between(xd, yd, max(yd), color='tab:orange', alpha=0.1)
plt.title("Train Data with Decision Boundary")
plt.show()


# In[21]:


rcParams['figure.figsize'] = 11.7,8.27
sns.scatterplot(data=test, x="MFCCs_10", y="MFCCs_17",hue='Species',palette=dict(HypsiboasCinerascens="#298CEE", HylaMinuta="#EE8C29"))
plt.plot(xd, yd, 'k', lw=1.5, ls='--')
plt.fill_between(xd, yd, min(yd), color='tab:blue', alpha=0.1)
plt.fill_between(xd, yd, max(yd), color='tab:orange', alpha=0.1)
plt.title("Test Data with Decision Boundary")
plt.show()


# In[22]:


rcParams['figure.figsize'] = 11.7,8.27
sns.scatterplot(data=df_sample, x="MFCCs_10", y="MFCCs_17",hue='Species',palette=dict(HypsiboasCinerascens="#298CEE", HylaMinuta="#EE8C29"))
plt.plot(xd, yd, 'k', lw=1.5, ls='--')
plt.fill_between(xd, yd, min(yd), color='tab:blue', alpha=0.1)
plt.fill_between(xd, yd, max(yd), color='tab:orange', alpha=0.1)
plt.title("Sample Data with Decision Boundary")
plt.show()


# In[23]:


df=pd.read_csv("Frogs.csv")


# In[24]:


df = df.sample(frac=1,random_state=42).reset_index(drop=True)


# In[25]:


trainpercent=0.80
df['Species_encoded']=df.Species.map({"HypsiboasCinerascens":0,"HylaMinuta":1})
train=df.sample(frac=trainpercent,random_state=102) #random state is a seed value
test=df.drop(train.index)


# In[26]:




X_train=train[['MFCCs_10', 'MFCCs_17']].to_numpy()

y_train=train[['Species_encoded']].to_numpy()
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))


X_test=test[['MFCCs_10', 'MFCCs_17']].to_numpy()

y_test=test[['Species_encoded']].to_numpy()
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
                           
                           


# In[27]:


logistic_completedata=GLm(2,1)


# In[28]:


learning_rate = 0.01                               
criterion = nn.BCELoss()
passes=8000
optimizer = torch.optim.SGD(logistic_completedata.parameters(), lr=learning_rate)

for epoch in range(passes):
#     print(epoch)
    y_pred = logistic_completedata(X_train)
    loss = criterion(y_pred, y_train)             
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % 500 == 0:                                         
       
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


# In[29]:


print("Training Accuracy")
y_predicted = logistic_completedata(X_train)
y_predicted_cls = y_predicted.round()
acc = y_predicted_cls.eq(y_train).sum() / float(y_train.shape[0])
print(f'accuracy: {acc.item():.4f}')


# In[30]:


print("Test Accuracy")
y_predicted = logistic(X_test)
y_predicted_cls = y_predicted.round()
acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
print(f'accuracy: {acc.item():.4f}')


# In[31]:


l = logistic_completedata.state_dict()
w1,w2=np.array(l['linear.weight'])[0]
w0=np.array(l['linear.bias'])[0]


c=-(w0/w2)
m=-(w1/w2)

x1min,x1max=df["MFCCs_10"].min(),df["MFCCs_10"].max()
x2min,x2max=df["MFCCs_17"].min(),df["MFCCs_17"].max()

xd = np.array([x1min, x1max])
yd = m*xd + c


# In[32]:


rcParams['figure.figsize'] = 11.7,8.27
sns.scatterplot(data=train, x="MFCCs_10", y="MFCCs_17",hue='Species',palette=dict(HypsiboasCinerascens="#298CEE", HylaMinuta="#EE8C29"))
plt.plot(xd, yd, 'k', lw=1.5, ls='--')
plt.fill_between(xd, yd, min(yd), color='tab:blue', alpha=0.1)
plt.fill_between(xd, yd, max(yd), color='tab:orange', alpha=0.1)
plt.title("Train Data with Decision Boundary")
plt.show()


# In[33]:


rcParams['figure.figsize'] = 11.7,8.27
sns.scatterplot(data=test, x="MFCCs_10", y="MFCCs_17",hue='Species',palette=dict(HypsiboasCinerascens="#298CEE", HylaMinuta="#EE8C29"))
plt.plot(xd, yd, 'k', lw=1.5, ls='--')
plt.fill_between(xd, yd, min(yd), color='tab:blue', alpha=0.1)
plt.fill_between(xd, yd, max(yd), color='tab:orange', alpha=0.1)
plt.title("Test Data with Decision Boundary")
plt.show()


# In[34]:


rcParams['figure.figsize'] = 11.7,8.27
sns.scatterplot(data=df, x="MFCCs_10", y="MFCCs_17",hue='Species',palette=dict(HypsiboasCinerascens="#298CEE", HylaMinuta="#EE8C29"))
plt.plot(xd, yd, 'k', lw=1.5, ls='--')
plt.fill_between(xd, yd, min(yd), color='tab:blue', alpha=0.1)
plt.fill_between(xd, yd, max(yd), color='tab:orange', alpha=0.1)
plt.title("Complete Data with Decision Boundary")
plt.show()


# In[ ]:





# In[ ]:




