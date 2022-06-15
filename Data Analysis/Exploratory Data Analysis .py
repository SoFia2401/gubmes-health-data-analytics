#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()


# In[4]:


#Let's get some general info 
df.info()


# In[5]:


#We can see we have equal number of non null values for all variables except bmi. 
#We will need to take away any null values.

# drop all rows with any NaN and NaT values
df1 = df.dropna()

#check info of new dataset 
df1.info()


# In[6]:


#Perfect now we have only non-null values, now let's get some overall statistics 
df1.describe()


# In[7]:


sizes = df1['stroke'].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')


# In[8]:


#Plot shows we have a lot more of patients that did not have stroke (0) than did have stroke (1)


# In[9]:


#Lets have a look at the gender distribution
sizes = df1['gender'].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')


# In[10]:


#We see as well that there are a lot more female patients than male


# In[16]:


#Lets have a look at the number of people with hypetension
sizes = df1['hypertension'].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')
plt.xlabel('Hypertension')


# In[18]:


#Lets have a look at the number of people with heart disease
sizes = df1['heart_disease'].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')
plt.xlabel('Heart disease')


# In[17]:


#Lets have a look at the number of people with stroke
sizes = df1['stroke'].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')
plt.xlabel('Stroke')


# In[28]:


list= df1[(df1['hypertension'] == 1) & (df1['heart_disease'] == 1) & (df1['stroke'] == 1)]
list


# In[43]:


#Of the people who have hypertension what percentage also has stroke and heart disease 
num_h_h_s=list['id'].count()
hypertension= df1[(df1['hypertension'] == 1)]
total_h= hypertension['id'].count()
percentage_h_h_s= num_h_h_s/total_h *100
percentage_h_h_s

