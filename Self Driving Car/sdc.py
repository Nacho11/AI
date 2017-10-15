# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# We can make how many ever NN we need using this class 
class Network(nn.Module): #Using inheritance to inherit all the tools from the nn.module class
    # this is the complete NN
    def __init__(self, number_of_input_neurons, number_of_actions): 
        #self refers to the object that will be created from this class
        #Whenever I want to use a variable from this object I will use self before the variable
        #to specify that this is a variable of the object
        #number of input neurons - describe the state
        #we can totally self drive with 3 sensors and the orientation
        # we can have 3 acctions - stored in number_of_actions
        super(Network, self).__init__() #inherits from the nn.module so that we can use everything from it
        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_actions = number_of_actions # output of output layer 
        # full connections between the different layers of the NN
        # After some experiments we can find out the total number of neurons in the hidden layer - 30
        self.full_conn_1 = nn.Linear(self.number_of_input_neurons, 30) # full connection btwn input layer and 1st layer
        self.full_conn_2 = nn.Linear(30, self.number_of_actions) #Num of neurons in the final output.
        
    #Forwards Propagation    
    def forward(self, state):
        x = Functional.relu(self.full_conn_1(state))
        q_values = self.full_conn_2(x)
        return q_values
    
#Implementing Experience replay
#What is Experience replay - All this is built on MDPs - events are know for going from
#one state to another - one timestamp is not enough to understand long term correlations 
#Instead of considering only the current state that is only one state at time, we are going to consider more in the past.
#So we put 100 last transactions into the memory. Thus deep Q learning is better.
        
class ReplayMemory(object):
    #Experience replay
    def __init__(self, capacity): # capacity will be 100 - last 100 transactions - max transactions
        self.capacity = capacity
        self.memory = [] #We'll put the last 100 transactions. 
        #Each time we reach a future state we'll put the transaction into the memory
        
        #Push function - To plant the events in this memory list and then we'll use 
        #the capacity - memory list always contains 100 elements.
        #Two tasks - append a new event in the memory and make sure only 100 events in the list
    def push(self, event): #Event has 4 elements - last state, new state, last action, last reward
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]                    
         
    #Sample - will get random samples from the memory of the last capacity elements
    def sample(self, batch_size):
        #*zip is like reshape function
        #If list - [[1,2,3], [4,5,6]], then zip(*list) = [(1,4), (2,5), (3,6)] 
        #Say 1 is the state, 2 is the action and 3 is the reward - we need a batch for
        #state1 and state2 , reward1 and reward2, action1 and action2
        samples = zip(*random.sample(self.memory, batch_size))#Will contain the samples from the memory
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #maps the samples to pytorch variable that will contain a tensor and a gradient  
        #get a list of batches each well aligned   

# Implementing the Deep Q learning
class Dqn(): # Deep Q Network
    
    def __init__(self, number_of_input_neurons, number_of_actions, gamma):
        self.gamma = gamma # delay coefficient
        self.reward_window = [] # sliding window of the mean of the last 100 words which you will use just
        # to evaluate the evolution of the performance   
        self.model = Network(number_of_input_neurons, number_of_actions)
        self.memory = ReplayMemory(100000)# we'll sample a small subset from this memory                   
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(number_of_input_neurons).unsqueeze(0) #The input has to always be in a batch - create a fake dimension corresponding the batch
        self.last_action = 0 #The last action 
        self.last_reward = 0 
        
        #Function that will select the right action each time.
    def select_action(self, state): #We will feed the input state into the neural network the one that we
        #build right above we get the outputs the Q values then using the soft max functions
        #we will get the final action to play
        #Idea of soft max is to get the best action to play at each time, but at the same time
        #we will be exploring the different actions.
        #Softmax will generate a distribution of probabilities for each of the q values 
        #This Q function is a function of the state and the action
        # 3 Q values - with 2 different prob which sum up to 1.
        #Alternative to this is RMX - taking the maximum of the Q values
        probs = Functional.softmax(self.model(Variable(state, volatile=True))*7) #Input the Q-values
        #State will be a torch tensor 
        # 7 is the temperature parameter - by increasing it our code will look more like a 
        #car rather than an insect, the higher the temperature, the higher will be the 
        # probability of the winning Q value. 
        # softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3]*3) = [0,0.02,0.98]  
        # By increasing the Q value the action will be clear
        action = probs.multinomial() # Will give us a random draw from the samples - 
        #returns a py torch variable with 
        return action.data[0,0] 
        
     # We will train the deep neural network 
    #takes batch_state, batch_next_state, batch_reward and finally batch action
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #We have to take these batches from the memory which become our transitions and 
        #then eventually we will get the different outputs for each of the states of the input states
        #and we will do this for the batch states and rest of the input mentioned above
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #Batch size and batch action should have the same size
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = Functional.smooth_l1_loss(outputs, target) # really good loss function
        self.optimizer.zero_grad()# WE must re initialize the adam optimizer at each iteration of the loop
        td_loss.backward(retain_variables = True) # retain_variable = true is to free memory.
        self.optimizer.step() # updates the weights
        
        #update func - will update when the AI will discover the new state
        #Updates everything once the AI reaches new state
        #This is the connection between the AI and the game class in map.py
    def update(self, reward, signal): #The last reward and last signal coming from map
        new_state = torch.Tensor(signal).float().unsqueeze(0)  #signal is a list of 5 variables - input to nn so should be tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #After reaching new state you need to update the memory - add the new transition
        #Self.last_action is a tensor which has only 1 value (0, 1, or 2), it is known as long tensor
        
        #We updated last state and memory done with one transition - we have to play a action
        action = self.select_action(new_state) # We play the action after reaching the new state
        #After selecting the action it is time to learn from the actions from the last 100 events
        #Need to add a is condition if the elements are above 100 transistions
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        #We have reached the new state but haven't updated the new state yet
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action #The action that was played when reaching the new state.
        
    #Next is to make a score function, save and load functions.
    def score(self):
        #Compute the mean of all the rewards in the reward window
        return sum(self.reward_window) / (len(self.reward_window)+1) # TO avoid dividing by 0
    
    #The save function so that we can save the model
    def save(self): #WE'll save the NN and the optimizer - we just want to save the last weights and we need optimizer as it is connected to the weights
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    }, 'last_brain.pth') #.Pth file contains the last version of state_dict and optimizer
        
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            #Update the model and optimizer with the values is last_brain.pth
            #Using the load state dict
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_state_dict(checkpoint['optimizer'])
            print "Completed"
        else:
            print "No checkpoint found"
            
            
        
        
        
        
        