#numpy is for working with arrays.
import numpy as np
#implementing the array with pytorch
import torch
#nn includes convolutional layers that will be used to train the AI.
import torch.nn as nn
#all the functions to be used in a neural network such as activation function, maxpooling etc...
import torch.nn.functional as F
#optimizer for AI. I'm using atom optimizer...
import torch.optim as optim
#imported for fast computation of AI. such as gradient descent
from torch.autograd import Variable
# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
#this includes action space for the specific game we'll play
#6 actions: move left, move right, move forward, turn left, turn right and shoot.
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience, image_preprocessing

#inherit nn.Module to use its functions
class Brain(nn.Module):

    #constructor of the brain.
    #number actions is necessary to play with the AI in other doom environments
    def __init__(self, number_actions):
        super(Brain, self).__init__()
        #apply convolution to input image
        #in_channels = input of the convolution is 1 because the images are in black and white
        #out_channels = number of features we want to detect in original images. 32 is common practice
        #32 means 32 processed images with each includes a specific feature
        #kernel_size is 5x5 for future detection. simply a matrix. then the size will be reduced to detect some more specific features
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        #output of convolution1 is input of conv2 layer
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        #we need 64 out channels in order to detect some more specific features of the surroundings
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        #in full connection, the vectors will be flattened.
        #full connection is a member of Linear class.
        #out features are number of neurons in hidden layer.
        self.fc1 = nn.Linear(in_features = self.countNeurons((1, 80, 80)), out_features = 40)
        #connection between hidden layer and output layer.
        #out features is number of out neurons and since each neuron is one q value and one q value corresponds to
        #one action to complete the game the out features will be number of actions that we need to perform
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)

    #number of output neurons only depend on the dimension of the image and
    #because of that, the function will only get the image dimensions to calculate
    #number of neuron in the network.
    def countNeurons(self, image_dim):
        #image dimension will be 80 x 80 to feed the network
        #create a fake image to count neurons and the pic is random because the neurons only depend on
        #dimensions not pixels.
        #convert it to a torch Variable to pass it into network
        x = Variable(torch.rand(1, *image_dim)) #create fake image
        #convert x to a convoluted image
        #apply max pooling to the image that is convoluted. 3 is common kernel size, 2(2x2) is how many pixels it is gonna stride for max pooling.
        #then apply rectifier activation function to activate the neurons in all pooled conv. layer (relu)
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        #take all the pixels of all channels and put one after the other to be inputted to the network.
        return x.data.view(1, -1).size(1)

    # function to forward the logic to body of the AI.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # flatten the channels in 3rd conv. layer to pass it to hidden layers.
        x = x.view(x.size(0), -1)
        # send the conv layers to the hidden layers and finally to output layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Making the body

class Body(nn.Module):
    
    def __init__(self, T):
        super(Body, self).__init__()
        self.T = T

    #turn signals from brain into actions
    #outputs = output from the brain.
    def forward(self, signalFromBrain):
        # distribution of probabilities with q values which is 7. also running is included.
        # the higher the T is, the less is the exploration of other actions will be done
        # because best action will already be selected
        probs = F.softmax(signalFromBrain * self.T)
        actions = probs.multinomial()
        return actions

# Making the AI

class AI:
    #assigning AI's brain and body to some variables.
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputImages):
        # receive input images from the game by converting the images into numpy arrays.
        # then convert numpy array into torch tensor
        # then convert this tensor to a torch variable which contains both the tensor and a gradient to be processed
        # numpy arrays with type float is created.(dtype)
        input = Variable(torch.from_numpy(np.array(inputImages, dtype = np.float32)))
        # apply the brain to input images
        output = self.brain(input)
        # brain returns the output and that will be forwarded into body
        actions = self.body(output)
        # actions have torch format so we need to convert them into numpy arrays
        return actions.data.numpy()


def initialize_environment():
    # use image_preprocessing class to make the dimensions of the image as desired
    environment = image_preprocessing.ImagePreprocessing(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomDeathmatch-v0"))), width = 80, height = 80, grayscale = True)
    environment = gym.wrappers.Monitor(environment, "videos", force = True)
    number_of_actions = environment.action_space.n
    return environment, number_of_actions

def initialize_brain_and_body(number_of_actions):
    convolutional_network = Brain(number_of_actions)  # build the brain
    body = Body(T=1.0)  # build the body
    ai = AI(brain=convolutional_network, body=body)  # combine both
    return convolutional_network, ai



doom_environment, number_of_actions = initialize_environment()

convolutional_network, ai = initialize_brain_and_body(number_of_actions)

# Building an AI


# Setting up Experience Replay
# instead of getting reward at each step AI get its reward after n_steps which is 10.
# so learning will take place in 10 transitions
n_steps = experience.NStepProgress(environment = doom_environment, intelligence = ai, n_steps = 10)
memory = experience.ReplayMemory(n_steps = n_steps, capacity = 10000) #  capacity keeps track of last 10000 steps performed by AI.
    
# Implementing Eligibility Trace

# function takes batch because we'll get some inputs and targets using a batch
def eligibility_trace(batch):
    # we need a gamma for n-step learning with eligibility trace
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch: # series of ten transitions.
        # put last and first input states into a numpy array then convert it into a torch variable
        # dtype is for converting into torch variable
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = convolutional_network(input)  # get output signal of the brain
        cumulative_reward = 0.0 if series[-1].done else output[1].data.max()  # start computing the cumulative reward.
        for step in reversed(series[:-1]):
            cumulative_reward = step.reward + gamma * cumulative_reward # update the cumulative reward by adding step reward
        # state object has four attributes: observation, reward, done and info. (https://gym.openai.com/docs/)
        state = series[0].state  # get the state of first transition
        target = output[0].data  # get the Q value of first state
        target[series[0].action] = cumulative_reward  # take the action corresponding to first step of the series
        inputs.append(state)  # append the state of the first step of the series into inputs array
        targets.append(target)  # append the target of the first step of the series into outputs array
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)  # return updated inputs and targets

# Making the moving average on 100 steps
class MA:
    # size of the list of the rewards of which the average will be computed.
    def __init__(self, size):
        self.list_of_rewards = []  # list containing the rewards
        self.size = size
    # function to add the cumulative reward to the list of rewards.
    def add(self, rewards):
        # when we get a new cumulative reward, it will add
        # sometimes the reward may be a list so we need to check the type of the reward.
        # if list, apply addition procedure, else append that reward into the list.
        if isinstance(rewards, list):
            self.list_of_rewards += rewards  # appending two lists together.
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size: # if the list size exceeds 100, delete the first element to make
            del self.list_of_rewards[0]              # sure that the list does not contain more than 100 elements.
    def average(self):  # function to compute the average of the lists elements.
        return np.mean(self.list_of_rewards)
ma = MA(100)

# training the AI

# loss is mean squared error for Q calculation
# for regression this is generally used.
loss = nn.MSELoss()

# first parameter is to make the connection between the optimizer and the parameters of our neural network
# which are the weights of the neurons of the brain.
# second parameter is learning rate and it is small because we don't want it to converge too fast
optimizer = optim.Adam(convolutional_network.parameters(), lr = 0.001)

#epoch is the number of trials that AI will have.
avg_reward = 0
epoch = 0
#iterate the loop until the bot reaches its goal...
while avg_reward < 20:
    epoch += 1
    memory.run_steps(200)  # 200 successive runs of 10 steps
    for batch in memory.sample_batch(128):  # series of 10 steps as opposed to before where the batches were some batches of single transitions.
        inputs, targets = eligibility_trace(batch) # apply eligibility trace to the batch
        inputs, targets = Variable(inputs), Variable(targets) # turn them into torch variables
        predictions = convolutional_network(inputs) # get the predictions for the actions
        loss_error = loss(predictions, targets) # compute loss by looking at the predictions and targets
        optimizer.zero_grad() # initialize the gradient descent method to back propagate the loss into neural network
        loss_error.backward()  # back propagate the error
        optimizer.step() # update the weights of the neural network
    # NSteps object from experience_replay file. Gets new cumulative rewards of the steps
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)  # this line will add step reward into moving average
    avg_reward = ma.average()  # using average method from moving average class
    print("Epoch: %s, Reward: %s" % (str(epoch), str(avg_reward)))  # printing the average reward at every iteration
    if avg_reward >= 20: # if AI gets above 2000 points we will assume that AI will win
        print("Congrats! The AI wins!!!")
        break

# Closing the Doom environment
doom_environment.close()