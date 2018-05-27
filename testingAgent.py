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



#How to run this file? $ python testingAgent.py path_of_dir_containing_model test_time_in_hours -i input_dir_containing_seeds_path -o output_dir_path [optional-other afl options like -d] binary_to_fuzz_path [options_for_binary_to_fuzz] @@ 
#This file contains code for TESTING and agent.py contains code for training.

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
argvType = c_char_p * (numberOfArguments -1)
argv = argvType()

#here the argIndex starts from 2 because we don't need to pass 0)testingAgent.py and 1)path_of_dir_containing_model to main function
argIndex = 2
while argIndex < numberOfArguments:
    argv[argIndex - 2] = c_char_p((sys.argv[argIndex]).encode('utf-8'))
    argIndex = argIndex + 1
argv[argIndex - 2]=None
argv[0] = c_char_p(('afl-fuzz').encode('utf-8'))

# Calling the main function which setup the fuzzer and does some pre-processing like executing
# the seed test-cases and setting up the virgin and trace_bits etc. 
adder.main((numberOfArguments - 2), argv);

#configurable parameters
sizeOfStateInbytes = 128
numberOfActions = 5
testingHours =  float(sys.argv[2])
#logTime: the time interval in min, after which we write the log about coverage in log.txt
logTime = 60
#fuzzingProb defines whether to fuzz the entire test case or just the State
fuzzingProb = 0.4
#iteration indicates the number of completed iterations in which we fuzz only the state
iteration = 0 

#result is the type of return from the getState function which returns the state of 128 bytes 
result = ndpointer(dtype=ctypes.c_uint8, shape=(sizeOfStateInbytes,))
get_state = adder.getState
get_state.restype = result
perform_action = adder.performAction
perform_action.restype = c_double
fuzzCompleteTestCase = adder.fuzz_one
return_coverage = adder.returnCoverage
return_coverage.restype = c_double
#define the action space 
adder.defineActionSpace()

#creates a file named log.txt in the output directory. The coverage will be logged in this file after every logTime minutes  
loc = str(sys.argv[6]) + "log.txt"
f= open(str(loc),"w+")
#We create a file reward.txt in the output directory which will contain the iteration number and the reward obtained in that iteration 
loc1 = str(sys.argv[6]) + "reward.txt"
file= open(str(loc1),"w+")

#location of the saved model
modelPath = str(sys.argv[1]) +"model.ckpt" 
metaGraphPath = str(sys.argv[1])+ "model.ckpt.meta"


#start a session
with tf.Session() as sess:
    e = 0.1
    #load the trained model
    saver = tf.train.import_meta_graph(metaGraphPath)
    saver.restore(sess,modelPath)
    print("Model restored.\n")
    graph = tf.get_default_graph()
    state_in = graph.get_tensor_by_name("state_in:0")
    weights = tf.trainable_variables()
    sess.run(weights)
    file.write("iteration, reward")
    testingTimeInMin = (testingHours * 60)     
    testing_start_time = time.time()
    timeSpentInTesting = 0
    while timeSpentInTesting < testingTimeInMin:  
        if np.random.rand(1) < fuzzingProb:
            #in this case fuzz the entire test case with the energy assigned by the AFL
            fuzzCompleteTestCase()
        else:    
            input_state = np.zeros(sizeOfStateInbytes)
            #get the state from AFL
            input_state = get_state()
            inputList = []
            for i in range(sizeOfStateInbytes):
                inputList.append(np.expand_dims(input_state[i],axis=0))
            inputArray = np.array(inputList, dtype=np.uint8)
            #we convert elements of a uint8 array into a binary-valued output array
            inputArray = np.unpackbits(inputArray, axis=1)
            #as we want 1 * 128 * 8 size matrix to be given as the input to LSTM, we perform expand_dims
            inputArray = np.expand_dims(inputArray,axis=0)
            
            #Now, access the operation that you want to run. 
            op_to_choseAction = graph.get_tensor_by_name("op_to_choseAction:0")
            #epsilon greedy policy
            if np.random.rand(1) < e:
                action = np.random.randint(numberOfActions)
            else:
                action = sess.run(op_to_choseAction,feed_dict={state_in:inputArray})
                  
            #Perform the selected action and get the reward and log the reward value 
            reward = perform_action(c_int(action))
            file.write("\n")
            file.write(str(iteration))
            file.write(",")
            file.write(str(reward))
            iteration = iteration + 1
        
        timeSpentInTesting = (time.time() - testing_start_time)/60 
        #write log after logTime minutes
        if timeSpentInTesting > logTime:
            f.write("\n *****The coverage after")
            f.write(str(logTime));
            f.write(" min:\n")
            f.write(str(return_coverage()))
            logTime = logTime + 60
#now write final coverage
f.write("\n The final coverage after testing:")
f.write(str(return_coverage()))
f.close()
file.close()
#this will call the function cleanUpThings() which perform the final cleanUp 
adder.cleanUpThings()
