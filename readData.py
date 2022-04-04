#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import glob


# In[71]:



path = r'./dataSet' 
all_files = glob.glob(path + "/*_tracks.csv")

frames = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df['fileId']=filename[8:10]
    frames.append(df)

bigFrame = pd.concat(frames, axis=0, ignore_index=True)


# In[78]:


bigFrame


# In[79]:


# TRACK FILE
BBOX = "bbox"
FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"
FILE_ID="fileId"


# In[80]:


grouped = bigFrame.groupby([TRACK_ID,FILE_ID], sort=False)
    # Efficiently pre-allocate an empty list of sufficient size
tracks = [None] * grouped.ngroups
current_track = 0
for group_id, rows in grouped:
    bounding_boxes = np.transpose(np.array([rows[X].values,
                                                rows[Y].values,
                                                rows[WIDTH].values,
                                                rows[HEIGHT].values]))
    tracks[current_track] = {TRACK_ID: np.int64(group_id),  # for compatibility, int would be more space efficient
                                 FRAME: rows[FRAME].values,
                                 BBOX: bounding_boxes,
                                 X_VELOCITY: rows[X_VELOCITY].values,
                                 Y_VELOCITY: rows[Y_VELOCITY].values,
                                 X_ACCELERATION: rows[X_ACCELERATION].values,
                                 Y_ACCELERATION: rows[Y_ACCELERATION].values,
                                 FRONT_SIGHT_DISTANCE: rows[FRONT_SIGHT_DISTANCE].values,
                                 BACK_SIGHT_DISTANCE: rows[BACK_SIGHT_DISTANCE].values,
                                 THW: rows[THW].values,
                                 TTC: rows[TTC].values,
                                 DHW: rows[DHW].values,
                                 PRECEDING_X_VELOCITY: rows[PRECEDING_X_VELOCITY].values,
                                 PRECEDING_ID: rows[PRECEDING_ID].values,
                                 FOLLOWING_ID: rows[FOLLOWING_ID].values,
                                 LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values,
                                 LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values,
                                 LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values,
                                 RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values,
                                 RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values,
                                 RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values,
                                 LANE_ID: rows[LANE_ID].values
                                 }
    current_track = current_track + 1
   





# In[81]:


tracks


# In[ ]:





# In[ ]:




