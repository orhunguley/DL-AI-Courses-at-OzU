# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:39:34 2018

@author: og4428
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:53:33 2018

@author: og4428
"""


# coding: utf-8

# In[121]:


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt 


# In[122]:


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion', one_hot=True)



# In[124]:


ind = 2342
plt.imshow(data.train.images[ind].reshape(28,28))
print('label:',np.argmax(data.train.labels[ind]))
plt.show()


# In[125]:


def var(name, shape, init=None, std=None):
    if init is None:
        if std is None:
            std = (2./shape[0])**0.5
        init = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable(name=name, shape=shape, 
                           dtype=tf.float32, initializer=init)


# In[126]:


g = tf.Graph()
with g.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28 ])
    Y = tf.placeholder(dtype=tf.float32, shape=[None,10])
    
    # hidden layer
    W1 = var('W1',[28*28, 500])
    b1 = var('b1',[500],tf.constant_initializer(0.1))
    out1 = tf.nn.relu(tf.matmul(X, W1) + b1)


    # hidden layer 2
    W2 = var('W2',[500, 1000])
    b2 = var('b2',[1000],tf.constant_initializer(0.1))
    out2 = tf.nn.relu(tf.matmul(out1, W2) + b2)
    
    # hidden layer 3
    W3 = var('W3',[1000, 250])
    b3 = var('b3',[250],tf.constant_initializer(0.1))
    out3 = tf.nn.relu(tf.matmul(out2, W3) + b3)


    #output layer
    W4 = var('W4',[250, 10])
    b4 = var('b4',[10],tf.constant_initializer(0.1))
    logits = tf.matmul(out3, W4) + b4
    
    #accuracy
    pred = tf.argmax(logits, axis=1)
    truth = tf.argmax(Y, axis=1)
    match = tf.cast(tf.equal(pred, truth), tf.float32)
    acc = tf.reduce_mean(match)
    

    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=Y))
    
    
    step = tf.train.AdamOptimizer().minimize(loss)


# In[127]:


sess = tf.InteractiveSession(graph=g)
tf.global_variables_initializer().run()


# In[128]:


bsize = 55
epoch_num=32
epoch_train=[]
epoch_test=[]

rows= data.train.num_examples

for epoch in range(epoch_num):
    for I in range(int(rows/bsize)):
        x, y = data.train.next_batch(bsize)
        _, acc_ = sess.run([step, acc], feed_dict={X:x, Y:y})
        #print('%d) acc: %2.4f' % (I,acc_), end='\r')


# In[129]:

    acc_train = sess.run(acc, feed_dict={X:data.train.images, Y:data.train.labels})    
    acc_test = sess.run(acc, feed_dict={X:data.test.images, Y:data.test.labels})
    
    epoch_train.append(acc_train)
    epoch_test.append(acc_test)
    
    
    print('\n', "Accuracy for test set in epoch number", epoch+1, "is : " , acc_train,end='\n')
    
    
    # In[130]:
    
    print("Accuracy for test set in epoch number", epoch+1, "is : ", acc_test ,end='\n')


e = np.arange(1,epoch_num + 1)
plt.plot(e,epoch_train)
plt.plot(e,epoch_test)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy %")

plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')

plt.show()

