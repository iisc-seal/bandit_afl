"""
Copyright 2018 Ketan Patil, Aditya Kanade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



#How to run this file?  $ python agent.py training_time_in_hours -i input_dir_containing_seeds_path -o output_dir_path [optional-other afl options like -d] binary_to_fuzz_path [options_for_binary_to_fuzz] @@ 
#This file contains code for TRAINING and testingAgent.py contains code for TESTING

from numpy.ctypeslib import ndpointer
from ctypes import *
import ctypes
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import time
import math
import sys
from decimal import Decimal

#load the shared object file
adder = CDLL('./fuzzer.so')
numberOfArguments = len(sys.argv)
#declaring an array of strings which will contain the parameters to be passed to the main function of afl-fuzz.c
argvType = c_char_p * numberOfArguments
argv = argvType()

argIndex = 1
while argIndex < numberOfArguments:
    argv[argIndex - 1] = c_char_p((sys.argv[argIndex]).encode('utf-8'))
    argIndex = argIndex + 1
argv[argIndex - 1] = None
argv[0] = c_char_p(('afl-fuzz').encode('utf-8'))



# Calling the main function which setup the fuzzer and does some pre-processing like executing
# the seed test-cases and setting up the virgin and trace_bits etc. 
adder.main((numberOfArguments - 1), argv);

#configurable parameters
sizeOfStateInbytes = 128
numberOfActions = 5
#trainingHours defines the time to be trained for 
trainingHours =  float(sys.argv[1])
learningRate = 0.001
#fuzzingProb defines whether to fuzz the entire test case or just the State  
fuzzingProb = 0.4
#iteration indicates the number of completed iterations in which we fuzz only the state
iteration = 0 

#result is the type of return from the getState function which returns the state of 128 bytes 
result = ndpointer(dtype=ctypes.c_uint8, shape=(sizeOfStateInbytes,))
#defining function pointer to the getState function 
get_state = adder.getState
get_state.restype = result

#defining function pointer to the performAction function
perform_action = adder.performAction
perform_action.restype = c_double

#defining function pointer to the fuzz_one function
fuzzCompleteTestCase = adder.fuzz_one
return_coverage = adder.returnCoverage
return_coverage.restype = c_double

#define the action space, this function will crate the array of multipliers  
adder.defineActionSpace()

#We create a file named log.txt in the output directory. The final coverage will be saved in this file. 
#Anyways this file is not much useful, as the afl also creates the file fuzzer_stats file in the output directory 
#which will contain the coverage and other information
loc = str(sys.argv[5]) + "log.txt"
f= open(str(loc),"w+")

#We create a file reward.txt in the output directory which will contain the iteration number and the reward obtained in that iteration 
loc1 = str(sys.argv[5]) + "reward.txt"
file= open(str(loc1),"w+")

#location to save our trained model
modelPath = str(sys.argv[5]) +"model.ckpt"

#clear the graph
tf.reset_default_graph() 

#our state is a matrix of 1 * 128 * 8
state_in = tf.placeholder(shape=[1,sizeOfStateInbytes, 8],dtype=tf.float32, name="state_in")

#We are using number of LSTM units to be equal to 100
nofLstmUnits = 100
cell = tf.contrib.rnn.LSTMCell(nofLstmUnits)
dropoutLstmLayer = 0.2 
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropoutLstmLayer)
_,hstate = tf.nn.dynamic_rnn(cell,state_in,dtype=tf.float32)
hstateTensor = tf.convert_to_tensor(hstate)
#we will just take take the final hidden state as the encoding of input state
finalHiddenState = hstateTensor[0][0]
finalHiddenState = tf.expand_dims(finalHiddenState,axis=0)
#we pass our encoding to our network, as an input state
output = slim.fully_connected(inputs = finalHiddenState,num_outputs = numberOfActions,weights_initializer=tf.initializers.random_uniform,activation_fn=tf.nn.softmax)
output = tf.reshape(output,[-1])
#choosing the action with maximum output value
chosen_action = tf.argmax(output,0,name="op_to_choseAction")

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32, name="reward_holder")
action_holder = tf.placeholder(shape=[1],dtype=tf.int32, name="action_holder")
responsible_weight = tf.slice(output,action_holder,[1])
#We use policy gradient method to update the weights
loss = -(tf.log(responsible_weight)*reward_holder)
#We use the gradient descent optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
update = optimizer.minimize(loss) 

#we have to train all the weights including weights of LSTM and of our network
weights = tf.trainable_variables()
#epsilon value to be used during epsilon-greedy method
e = 0.1 
init = tf.global_variables_initializer()
#saver is for saving model after training
saver = tf.train.Saver()

#start a session 
with tf.Session() as sess:
    trainingTimeInMin = 60 * trainingHours
    timeSpentInTraining = 0
    start_time = time.time()
    sess.run(init)
    file.write("iteration, reward")
    #TRAINING PART
    while timeSpentInTraining < trainingTimeInMin:
        if np.random.rand(1) < fuzzingProb:
            #in this case fuzz the entire test case with the energy assigned by the AFL
            fuzzCompleteTestCase()
        else: 
            #in this case we will only fuzz the state   
            input_state = np.zeros(sizeOfStateInbytes)
            #get the state of size 128 unsigned bytes 
            input_state = get_state()
            inputList = []
            for i in range(sizeOfStateInbytes):
                inputList.append(np.expand_dims(input_state[i],axis=0))
            inputArray = np.array(inputList, dtype=np.uint8)
            #we convert elements of a uint8 array into a binary-valued output array
            inputArray = np.unpackbits(inputArray, axis=1)
            #as we want 1 * 128 * 8 size matrix to be given as the input to LSTM, we perform expand_dims
            inputArray = np.expand_dims(inputArray,axis=0)
            #epsilon greedy policy
            if np.random.rand(1) < e:
                action = np.random.randint(numberOfActions)
                #choose a random action
            else:
                action = sess.run(chosen_action,feed_dict={state_in:inputArray})
                #choose action predicted by our model
                
        
            #Perform the selected action and get the reward and log the reward value 
            reward = perform_action(c_int(action))
            file.write("\n")
            file.write(str(iteration))
            file.write(",")
            file.write(str(reward))
            iteration = iteration + 1
           
            #Update the network.
            feed_dict={reward_holder:[reward],action_holder:[action],state_in:inputArray}
            _,ww, my_output, my_responsibleWeight = sess.run([update,weights, output,responsible_weight], feed_dict=feed_dict)
             
        #update the time spent 
        timeSpentInTraining = (time.time() - start_time)/60
        
      
    #save this trained model once the training is finished
    save_path = saver.save(sess, modelPath)
    print("Model saved in path: %s" % save_path)
        
#now write coverage after training part
f.write("\n The final coverage after training:")
f.write(str(return_coverage()))
f.close()
file.close()
#this will call the function cleanUpThings() which perform the final cleanUp 
adder.cleanUpThings()
