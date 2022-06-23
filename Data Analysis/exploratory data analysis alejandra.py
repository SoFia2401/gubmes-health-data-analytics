import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()
df.info()

df1 = df.dropna()
df1.info()

#variable ever married: yes or no
sizes = df1["ever_married"].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')
plt.xlabel('Ever married')
plt.ylabel('Counts')
plt.title('Ever married')

#variable work type: private, self employed or other
sizes = df1["work_type"].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')
plt.xlabel('Work type')
plt.ylabel('Counts')
plt.title('Work type')

#variable residence type: urban or rural
sizes = df1["Residence_type"].value_counts(sort = True)
ax = plt.figure()
sizes.plot(kind='bar')
plt.xlabel('Residence type')
plt.ylabel('Counts')
plt.title('Residence type')

#visulise type of residence of people who suffer from stroke
totalpeople = df1['id'].count()
combine = df1[['stroke', 'Residence_type']].to_numpy()
urban = 0
for i in range(0,int(totalpeople)):
    if combine[i,0] == 1 and combine[i,1] == 'Urban':
        urban = urban + 1
rural = 0
for j in range(0,int(totalpeople)):
    if combine [j,0] == 1 and combine[j,1] == 'Rural':
        rural = rural + 1
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
res = ['Urban', 'Rural']
count = [urban, rural]
ax.bar(res, count)
plt.xlabel('Residence type')
plt.ylabel('Counts')
plt.title('Type of residence of people who suffer from stroke')
plt.show()