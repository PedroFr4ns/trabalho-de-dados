#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Health_Risk_Dataset.csv')
df


# In[3]:


df = df.drop(columns=['Patient_ID'])
df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[19]:


df = pd.read_csv('Health_Risk_Dataset.csv')
df


# In[20]:


df = df.drop(columns=['Patient_ID'])
df


# In[21]:


print(df["On_Oxygen"].unique())
print(df["On_Oxygen"].dtype)


# In[26]:


df["On_Oxygen"] = df["On_Oxygen"].astype(bool)
df


# In[27]:


df['O2_Scale'].value_counts()


# In[28]:


df["O2_Scale"] = df["O2_Scale"].map({1: False, 2: True}).astype(bool)
df


# In[29]:


df.info()


# In[30]:


print(df["Consciousness"].unique())
print(df["Consciousness"].dtype)


# In[31]:


list(df['Consciousness'].unique())

df['Consciousness'] = df['Consciousness'].map({'A': 1, 'P': 2, 'V': 3, 'U': 4, 'C': 5})
df


# In[33]:


df['Consciousness'] = df['Consciousness'].astype('category')
df


# In[34]:


print(df["Risk_Level"].unique())
print(df["Risk_Level"].dtype)


# In[35]:


list(df['Risk_Level'].unique())

df['Risk_Level'] = df['Risk_Level'].map({'Normal': 1, 'Low': 2, 'Medium': 3, 'High': 4}).astype('category')
df


# In[36]:


df.info()


# In[41]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# In[42]:


df.info(memory_usage = 'deep')


# In[44]:


df['Respiratory_Rate'] = df['Respiratory_Rate'].astype('int16')
df['Oxygen_Saturation'] = df['Oxygen_Saturation'].astype('int16')
df['Systolic_BP'] = df['Systolic_BP'].astype('int16')
df['Heart_Rate'] = df['Heart_Rate'].astype('int16')

df['Temperature'] = df['Temperature'].astype('float32')

df.info(memory_usage = 'deep')


# In[ ]:




