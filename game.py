import gym
import numpy as np
import random
import math



def train_Qtable(Q, num_episodes, epsilon, gamma, lr_rate):
    '''
    '''
	Q_old = Q.copy()
	for i in range(num_episodes):
        # define initial state
		state = env.reset()
		done = False
		while done == False:
			# First we select an action:
			if random.uniform(0, 1) < epsilon: # take a random number
				action = env.action_space.sample() # Explore action space
			else:
				action = np.argmax(Q[state,:]) # Exploit learned values
			# Then we perform the action and receive the feedback from the environment
			new_state, reward, done, info = env.step(action)
			# Finally we learn from the experience by updating the Q-value of the selected action
			update = reward + (gamma*np.max(Q[new_state,:])) - Q[state, action]
			Q[state,action] += learning_rate*update 
			if (Q_old == Q).all():
				print("Q table has been converged to optimal in {}th iteration ".format(i))
				return Q
				
			Q_old = Q.copy()
			state = new_state
    
    # even if Q table will not converge to optimal return latest updated Q table
    return Q


if __name__ == "__main__":

    # init game parameters
    num_episodes = 15000 
    gamma = 0.95
    learning_rate = 0.7 
    epsilon = 0.3

    # load game and reset to initial state
    env = gym.make("Taxi-v3")
    env.reset()
    # env.render()


    # initialize the Q table, states -> actions
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    
