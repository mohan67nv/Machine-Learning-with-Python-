"""
Importing the necessary libraries and packages for the task
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import os
import numpy as np
from sklearn import preprocessing
"""
os.environ is to remove the errors that are occuring
logs_path: the path to save the tensorboard file
"""
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
logs_path = 'H:/tmp/tensorflow_logs/example2'
"""
Defining the parameters learning_rate, epochs and the batch_size for the data
"""
learning_rate=0.01
training_epochs=200
batch_size=100
data=fetch_olivetti_faces() #loading the data
targets=data.target#extracting the targets
le=preprocessing.LabelBinarizer()#Using the label binarizer to encode the targets
targets=le.fit_transform(targets)
data = data.images.reshape((len(data.images), -1))#changing the data shape to 2D from 3D
x_train,x_test,y_train,y_test=train_test_split(data,targets,test_size=0.10)#splitting the data set into train and test

#Input graph for the tensorflow the faces data set is of 64*64  and 40 targets
x=tf.placeholder(tf.float32,[None,4096],name='InputData')#64*64=4096
y=tf.placeholder(tf.float32, [None,40],name='LabelData')#
W=tf.Variable(tf.zeros([4096,40]),name='Weights')#weights for the model
b=tf.Variable(tf.zeros([40]),name='Bias')#bias for the model
with tf.name_scope("Model"):
    prediction=tf.nn.softmax(tf.matmul(x,W)+b)#soft max function for the model
with tf.name_scope("Loss"):
    cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction),reduction_indices=1))#cross entropy for loss
with tf.name_scope("SGD"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#SGD for training
with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))#defining the accuracy function
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init=tf.global_variables_initializer()#initializing all the variables

tf.summary.scalar("loss", cost)#create a summary to track the cost
tf.summary.scalar("accuracy", accuracy)#Create a summary to track the accuracy
merged_summary_op = tf.summary.merge_all()#Merge all summaries into a single OP
with tf.Session() as sess:
    sess.run(init)#Running the initializer
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())#writing the logs to tensorboard
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(len(x_train)/batch_size)
        for i in range(total_batch):
            _,c,summary=sess.run([optimizer,cost,merged_summary_op], feed_dict={x:x_train,y:y_train})
            summary_writer.add_summary(summary,total_batch*epoch+i)#writing the log at every iteration
            avg_cost+=c/total_batch#computing the average loss

        print("Epoch:",epoch,"cost =",avg_cost)
    print("Training accuracy",accuracy.eval({x:x_train,y:y_train})*100)#train accuracy
    print("Test accuracy",accuracy.eval({x:x_test,y:y_test})*100)#test accuarcy

    print("Run the command line:\n" \
          "--> tensorboard --logdir ",logs_path)
