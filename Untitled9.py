
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import merapyar


# In[5]:

def binary_cross_entropy(x, z):
    #here x is the expected output of network and z is the actual output
    #sml is a small value to prevent log(0) error
    sml = 1e-12
    #now we calculate the loss
    return (-(x * np.log(z + sml) + (1. - x) * np.log(1. - z + sml)))


# In[7]:

np.random.seed(122)


# In[8]:

def arctan(x):
    return np.arctan(x)
def d_arctan(x):
    return 1 / (1 + x ** 2)


# In[ ]:

def Discriminator(img_in, reuse=None, prob=keep_prob,ksize = 5):
    

