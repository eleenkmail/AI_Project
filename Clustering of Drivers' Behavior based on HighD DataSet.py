#!/usr/bin/env python
# coding: utf-8

# ## import libraries

# In[49]:


import pandas as pd
import numpy as np
import math
import glob
from statsmodels import robust
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing


# ## Reading Tracks csv files and collect data in one data frame

# In[2]:



path = r'../dataSet' 
all_files = glob.glob(path + "/*_tracks.csv")

frames = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    if filename[8:9] == '0':
        df['fileId']=filename[12:13]
    else:    
        df['fileId']=np.int64(filename[11:13])
    frames.append(df)

bigFrame = pd.concat(frames, axis=0, ignore_index=True)


# In[127]:


#Naming features of data set 

FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"


# In[128]:


#Grouping each driver data by compound ID -track id and file id-
grouped = bigFrame.groupby([TRACK_ID,FILE_ID], sort=False)
    # Efficiently pre-allocate an empty list of sufficient size
tracks = [None] * grouped.ngroups
current_track = 0
for group_id, rows in grouped:
  
    tracks[current_track] = {TRACK_ID: (group_id),  # for compatibility, int would be more space efficient
                                 FRAME: rows[FRAME].values, 
                                
                                 X_VELOCITY: rows[X_VELOCITY].values,
                                 Y_VELOCITY: rows[Y_VELOCITY].values,
                                 X_ACCELERATION: rows[X_ACCELERATION].values,
                                 Y_ACCELERATION: rows[Y_ACCELERATION].values,
                                
                                 }
    current_track = current_track + 1
    
tracks=pd.DataFrame(tracks)


# ## DV1 & 2: Standard deviation for velocity and acceleration

# In[53]:


def DV_1_2(velocity):
    return velocity.std()


# In[54]:


velocity_std=map(DV_1_2, tracks['xVelocity'])
velocity_std=list(velocity_std)


# In[55]:


# add DV1 feature to the dataset.
tracks['DV1']=velocity_std


# In[56]:


longitudinal_std=map(DV_1_2, tracks['xAcceleration'])
longitudinal_std=list(longitudinal_std)


# In[57]:


# add DV2 feature to the dataset.
tracks['DV2']=longitudinal_std


# ## DV3 & 4 & 5: Coeffient of variance for velocity, acceleration and decleration

# ### DV3  -Coeffient of variance for velocity-

# In[58]:


def v_mean(velocity):
    return velocity.mean()

V_mean=map(v_mean,tracks['xVelocity'])
V_mean=list(V_mean)


# In[59]:


def DV3(velocity_std,velocity_mean):
    return (velocity_std/velocity_mean)*100


# In[60]:


Coeff_Variance_Velocity=map(DV3, tracks['DV1'], V_mean)
Coeff_Variance_Velocity=list(Coeff_Variance_Velocity)


# In[61]:


# add DV3 feature to the dataset.
tracks['DV3']=Coeff_Variance_Velocity


# ### DV4  -Coeffient of variance for acceleration-

# In[62]:


def DV4(track_acc):
    track_acc_=track_acc[track_acc>=0]
    count=len(track_acc_)
    if(count==0 or track_acc_.mean()==0):
        return 0
    
    return track_acc_.std() / track_acc_.mean()
    


# In[63]:


Coeff_Variance_Acceleration=map(DV4,tracks['xAcceleration'])
Coeff_Variance_Acceleration=list(Coeff_Variance_Acceleration)


# In[64]:


# add DV4 feature to the dataset.
tracks['DV4']=Coeff_Variance_Acceleration


# ### DV5  -Coeffient of variance for decleration-

# In[65]:


def DV5(track_dec):
    track_dec_=track_dec[track_dec<0]
    count=len(track_dec_)
    if(count==0 or track_dec_.mean()==0):
        return 0
    
    return track_dec_.std() / track_dec_.mean()
    


# In[66]:


Coeff_Variance_Decleration=map(DV5,tracks['xAcceleration'])
Coeff_Variance_Decleration=list(Coeff_Variance_Decleration)


# In[67]:


# add DV5 feature to the dataset.
tracks['DV5']=Coeff_Variance_Decleration


# ## DV6 & 7: Mean absolute deviation of speed and acceleration

#  ### DV6 -Mean absolute deviation of speed-

# In[68]:


def DV6(velocity):
    return robust.mad(velocity)


# In[69]:


velocity_mad=map(DV6,tracks['xVelocity'])
velocity_mad=list(velocity_mad)


# In[70]:


# add DV6 feature to the dataset.
tracks['DV6']=velocity_mad


# ### DV7 -Mean absolute deviation of acceleration-

# In[71]:


def DV7(track_acc):
    track_acc_=track_acc[track_acc>=0]
    count=len(track_acc_)
    if(count==0):
        return 0
    return robust.mad(track_acc_)


# In[72]:


acceleration_mad=map(DV7,tracks['xAcceleration'])
acceleration_mad=list(acceleration_mad)


# In[73]:


# add DV7 feature to the dataset.
tracks['DV7']=acceleration_mad


# ## DV8 & 9 & 10: Quantile coefficient of variation of normalised speed, acceleration and deceleration

# ### DV8 -Quantile coefficient of variation of normalised speed-

# In[74]:


def DV8(track_velocity):
    Q3=np.percentile(track_velocity, 75)
    Q1=np.percentile(track_velocity, 25)
    return (Q3 - Q1)*100 / (Q3 + Q1)


# In[75]:


Q_C_V=map(DV8,tracks['xVelocity'])
Q_C_V=list(Q_C_V)


# In[76]:


# add DV8 feature to the dataset.
tracks['DV8']=Q_C_V


# ### DV9 -Quantile coefficient of variation of normalised acceleration-

# In[77]:


def DV9(track_acc):
    track_acc_=track_acc[track_acc>=0]
    count=len(track_acc_)
    if(count==0):
        return 0
    
    Q3=np.percentile(track_acc_, 75)
    Q1=np.percentile(track_acc_, 25)
    return ((Q3 - Q1) / (Q3 + Q1))*100


# In[78]:


Q_C_A=map(DV9,tracks['xAcceleration'])
Q_C_A=list(Q_C_A)


# In[79]:


# add DV9 feature to the dataset.
tracks['DV9']=Q_C_A


# ### DV10 -Quantile coefficient of variation of normalised deceleration-

# In[80]:


def DV10(track_dec):
    track_dec_=track_dec[track_dec<0]
    count=len(track_dec_)
    if(count==0):
        return 0
    
    Q3=np.percentile(track_dec_, 75)
    Q1=np.percentile(track_dec_, 25)
    return (Q3 - Q1)*100 / (Q3 + Q1)


# In[81]:


Q_C_D=map(DV10,tracks['xAcceleration'])
Q_C_D=list(Q_C_D)


# In[82]:


# add DV10 feature to the dataset.
tracks['DV10']=Q_C_D


# ## DV11 & 12 & 13: Percentage of time the mean normalised

# ### DV11 -for Velocity-

# In[83]:


def DV11(track_velocity, track_dv1):
    margin_of_error= track_dv1*2
    mean= track_velocity.mean()
    interval= margin_of_error + mean
    summation=track_velocity[track_velocity>=interval].sum()
    Percentage= 100*summation / len(track_velocity)
    return Percentage
    
    


# In[84]:


dv11=map(DV11,tracks['xVelocity'], tracks['DV1'])
dv11=list(dv11)


# In[85]:


# add DV11 feature to the dataset.
tracks['DV11']=dv11


# ### DV12 -for Acceleration-

# In[86]:


def DV12(track_acc, track_dv2):
    track_acc_=track_acc[track_acc>=0]
    count=len(track_acc_)
    if count==0:
        return 0
    
    margin_of_error= track_dv2*2
    mean= track_acc_.mean()
    interval= margin_of_error + mean
    summation=track_acc_[track_acc_>=interval].sum()
    Percentage= 100*summation / count
    return Percentage


# In[87]:


dv12=map(DV12,tracks['xAcceleration'], tracks['DV2'])
dv12=list(dv12)


# In[88]:


# add DV12 feature to the dataset.
tracks['DV12']=dv12


# ### ### DV13 -for Deceleration-

# In[89]:


def DV13(track_dec, track_dv2):
    track_dec_=track_dec[track_dec<0]
    count=len(track_dec_)
    if count==0:
        return 0
    
    margin_of_error= track_dv2*2
    mean= track_dec_.mean()
    interval= margin_of_error + mean
    summation=track_dec_[track_dec_>=interval].sum()
    Percentage= 100*summation / count
    return Percentage


# In[90]:


dv13=map(DV13,tracks['xAcceleration'], tracks['DV2'])
dv13=list(dv13)


# In[94]:


# add DV13 feature to the dataset.
tracks['DV13']=dv13


# ## Clistering  

# In[95]:



# cleaned_data = tracks.replace([np.inf, -np.inf], np.nan)
# cleaned_data= cleaned_data.dropna().reset_index()


# In[115]:


#COPY the DVs COLUMNS TO NEW DATA FRAME TO DO CLUSTER FOR IT.

selected_columns = tracks[["DV1","DV2","DV3","DV4","DV5","DV6","DV7","DV8","DV9","DV10","DV11","DV12","DV13"]]
measures = selected_columns.copy()

# REMOVE OUTLIERS -INFINTE VALUES-
cleaned_data = measures.replace([np.inf, -np.inf], np.nan)
cleaned_data= cleaned_data.dropna()

# SCALING THE DATA WITH STANDARDIZATION SCALE
scaler = preprocessing.StandardScaler().fit(cleaned_data)
scaled_measures = scaler.transform(cleaned_data)

scaled_measures=pd.DataFrame(scaled_measures)


# ## DONING CLUSTERING WITH NUMBER 3 OF CLUSTERS USING KMEANS
# 

# In[116]:


kmeans_3= KMeans(n_clusters=3, random_state=0,init="k-means++")
kmeans_3.fit(scaled_measures)


# In[120]:


# PUT THE CLUSTERED DATA IN DATA FRAME AND LABELED THE CLUSTERS THEN GROUP THE DRIVERS WITH ITS LABEL

clusters_3=pd.DataFrame(scaled_measures,columns=selected_columns.columns)
clusters_3['label']=kmeans_3.labels_

#CENTERS OF THE CLUSTERS
centers_3=kmeans_3.cluster_centers_


#COUNT OF EACH LABEL

labels_count_3=clusters_3.groupby('label').size().rename('count')
labels_count_3=pd.DataFrame(labels_count_3)


#HYPOTHESIS OF LABELS

labels_count_3.plot.bar()
plt.show()


# ## DONING CLUSTERING WITH NUMBER 2 OF CLUSTERS USING KMEANS
# 

# In[121]:


kmeans_2= KMeans(n_clusters=2, random_state=0,init="k-means++")
kmeans_2.fit(scaled_measures)


# In[126]:


# PUT THE CLUSTERED DATA IN DATA FRAME AND LABELED THE CLUSTERS THEN GROUP THE DRIVERS WITH ITS LABEL

clusters_2=pd.DataFrame(scaled_measures,columns=selected_columns.columns)
clusters_2['label']=kmeans_2.labels_

#CENTERS OF THE CLUSTERS

centers_2=kmeans_2.cluster_centers_
centers_2=pd.DataFrame(centers_2)


#COUNT OF EACH LABEL

labels_count_2=clusters_2.groupby('label').size().rename('count')
labels_count_2=pd.DataFrame(labels_count_2)



#HYPOTHESIS OF LABELS

labels_count_2.plot.bar()
plt.show()

