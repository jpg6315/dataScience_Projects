#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Keras is an api Lego-like building block for building and defining models.tf.data is an easy input pipeline
Eager execution makes TensorFlow feel like regular Python


# <img align="left" width="300" height="2000" src="Clustering.png">
# &nbsp;                                                                                                                            
# &nbsp;
# 
# # <center> <h1>K-MEANS CLUSTERING ANALYSIS</h1> </center> 
# #### <center> <h1>in JULIA</h1> </center>
# &nbsp;                                                                                                                            
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;                                                                                                                            
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp;
# &nbsp; 
# by Jesse P. Gutierrez Jr
#    UHD, Data Science

# In[ ]:





# In[ ]:





# In[1]:


import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


# In[ ]:




