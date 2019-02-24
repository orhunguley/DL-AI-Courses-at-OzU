
# coding: utf-8

# In[7]:


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt 


# In[8]:


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion', one_hot=True)


# In[9]:


ind = 1
plt.imshow(data.train.images[ind].reshape(28,28))
print('label:',np.argmax(data.train.labels[ind]))
plt.show()


# In[10]:


def var(name, shape, init=None, std=None):
    if init is None:
        if std is None:
            std = (2./shape[0])**0.5
        init = tf.truncated_normal_initializer(stddev=std)
    return tf.get_variable(name=name, shape=shape, 
                           dtype=tf.float32, initializer=init)

def conv(X, f, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(X, f, strides, padding)

def max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(X, ksize, strides, padding)


# In[11]:


g = tf.Graph()
with g.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    I = tf.reshape(X, [-1, 28,28,1])
    
    #Conv Layer 1
    W1 = var('W1', [5,5, 1, 16])
    b1 = var('b1', [16])
    out1 = tf.nn.relu(conv(I, W1) + b1)
    out1 = max_pool(out1)
    
    #Conv Layer 2
    W2 = var('W2', [5,5, 16, 32])
    b2 = var('b2', [32])
    out2 = tf.nn.relu(conv(out1, W2) + b2)
    out2 = max_pool(out2)
    
    flatten = tf.reshape(out2, [-1, 7*7*32])
    
    #Fully Connected Layer
    W3 = var('W3',[7*7*32, 500])
    b3 = var('b3',[500])
    out3 = tf.nn.relu( tf.matmul(flatten ,W3) + b3 )
    
    # Output layer
    W4 = var('W4',[500,10])
    b4 = var('b4',[10])
    logits = tf.matmul(out3, W4) + b4
    
    pred = tf.argmax(logits, axis=1)
    truth = tf.argmax(Y, axis=1)
    acc = tf.reduce_mean(
        tf.cast(tf.equal(pred,truth),tf.float32))

    s_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y)
    loss = tf.reduce_mean(s_loss)
    
    step = tf.train.AdamOptimizer().minimize(loss)


# In[12]:


sess = tf.InteractiveSession(graph=g)
tf.global_variables_initializer().run()



bsize = 55
epoch_num=32
epoch_train=[]
epoch_test=[]

rows= data.train.num_examples

for epoch in range(epoch_num):
    for I in range(int(rows/bsize)):
        x, y = data.train.next_batch(bsize)
        _, l,a = sess.run([step,loss, acc], feed_dict={X:x, Y:y})
        #print('%d) acc: %2.4f' % (I,acc_), end='\r')


# In[129]:

    acc_train = (sess.run(acc, feed_dict={X:data.train.images, Y:data.train.labels})) 
    acc_test = (sess.run(acc, feed_dict={X:data.test.images, Y:data.test.labels}))
    
    #epoch_train.append(acc_train)
    epoch_test.append(acc_test)
    
    
    #print('\n', "Accuracy for test set in epoch number", epoch+1, "is : " , acc_train,end='\n')
    
    
    # In[130]:
    
    print("Accuracy for test set in epoch number", epoch+1, "is : ", acc_test ,end='\n')


e = np.arange(1,epoch_num + 1)
#plt.plot(e,epoch_train)
plt.plot(e,epoch_test)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy %")

plt.legend(['Test Accuracy'], loc='lower right')

plt.show()



