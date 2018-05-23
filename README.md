# README #


# What is this repository for? 

Greybox Fuzzing As Contextual Bandits Problem:
This repository contains the code to formulate the energy prediction in greybox fuzzing as contextual bandits problem. We select 128 bytes from the test case and treat it as a state. We predict actions according to a neural network model given the input state, here the action space is the collection of multipliers of the energy value, to be given to the test case. We fuzz the state with the modified energy value. This tool is an extension of [American Fuzzy Lop (AFL)](http://lcamtuf.coredump.cx/afl/)

# How do I get set up?
Follow the instructions given below. A concrete example of fuzzing nm-new binary from the binutils is provided at the end.  
1)Install [Tensorflow CPU](https://www.tensorflow.org/install/)   
2)Clone this repository and change your current working directory to this directory.   
3)Compile the source code  
```bash
make clean all
```   
4)Now compile the target program(e.g. binutils) with afl-gcc. e.g.
```bash
CC=<path-to-bandit_afl-repository>/afl-gcc ./configure --disable-shared
make clean all   
```
5)Create an input directory containing some seed inputs, and 2 output directories, one for saving the training results and other for saving test results .
Start the training:   
```bash
python <path-to-bandit_afl-repository>/agent.py training_time_in_hours -i path_to_input_dir_containing_seeds -o path_to_first_output_dir [optional-other afl options like -d/-x etc] path_to_target_binary_to_fuzz [options_for_binary_to_fuzz]   
```
The above command creates an output directory and saves the trained model in the first output directory.   
6)Once the training is completed, start the testing.   
```bash
python <path-to-bandit_afl-repository>/testingAgent.py path_to_first_output_dir test_time_in_hours -i path_to_input_dir_containing_seeds -o path_to_second_output_dir [optional-other afl options like -d/-x etc] path_to_target_binary_to_fuzz [options_for_binary_to_fuzz]
```

# A concrete example: Fuzzing nm-new binary from binutils-2.26
1)Install [Tensorflow CPU](https://www.tensorflow.org/install/), if you haven't installed yet.   
2)Clone this repository.  
3)Change your current working directory to this repository and compile it using following commands  
```bash
cd <path-to-bandit_afl-repository>
make clean all
``` 
3)Download the source for binutils-2.26 to any location of your choice. 
```bash
wget https://ftp.gnu.org/gnu/binutils/binutils-2.26.tar.gz
```
4)Extract the binutils 
```bash
tar -xvzf ./binutils-2.26.tar.gz
```
5)Change your current directory to the extracted binutils folder
```bash
cd ./binutils-2.26/
```
6)Compile binutils with afl-gcc 
```bash
CC=<path-to-bandit_afl-repository>/afl-gcc ./configure --disable-shared
make clean all
```
NOTE: Before compiling binutils, make sure that you have all the binutils-2.26 prerequisites installed.   
7)Create an input directory and 2 output directories 
```bash 
#create an input directory
mkdir afl_in
#create output directories 
mkdir afl_train_out
mkdir afl_test_out
```
8)We use seed provided by the afl as an input seed for this experiment. 
```bash
#copy the seed provided by the afl to out input directory
cp <path-to-bandit_afl-repository>/testcases/others/elf/small_exec.elf ./afl_in
```
9)Change your current working directory back to bandit_afl   
10)Start training: here we are training our model for 0.5 hour and we use nm-new (with -C option) as a target binary to fuzz. You can change the training time and target binary as per your choice.  
```bash
python agent.py 0.5 -i <path-to-binutils-2.26-directory>/afl_in/ -o <path-to-binutils-2.26-directory>/afl_train_out/ <path-to-binutils-2.26-directory>/binutils/nm-new -C @@
```
NOTE: If you get any warnings related to core_patterns or cpu scaling governor then follow the instructions provided by the AFL.  
11)Once the training is finished, start testing: Here we are testing for 4 hours. You can change the test time as per your choice
```bash
python testingAgent.py <path-to-binutils-2.26-directory>/afl_train_out/ 4 -i <path-to-binutils-2.26-directory>/afl_in/ -o <path-to-binutils-2.26-directory>/afl_test_out/ <path-to-binutils-2.26-directory>/binutils/nm-new -C @@
```
12)All the results can be found in afl_test_out directory.
