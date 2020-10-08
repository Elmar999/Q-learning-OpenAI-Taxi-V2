import gym
import numpy as np
import random
import math



def train_Qtable(Q, env, num_episodes, epsilon, gamma, lr_rate):
    '''
    function trains Q table with given parameters
    Args:
        Q (numpy array): Q table which will be updated
        env (gym environment)
        num_episodes (int): number of games that will be played during training
        epsilon (int): probability threshold
        gamma (int): discount rate
        lr_rate (int): learning rate
    Returns:
        Q_optimal (numpy array): updated Q table which is converged to optimal
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


def launch_game(Q, env):
    # Is our Q good enough to guide us from start to goal without falling into the water?
    state = env.reset()
    env.render()
    done = False
    while done == False:
        # Take the action (index) with the maximum expected discounted future reward given that state
        action = np.argmax(Q[state,:])
        state, reward, done, info = env.step(action)
        env.render()


if __name__ == "__main__":

    # init game parameters
    num_episodes = 15000 
    gamma = 0.95
    learning_rate = 0.1
    epsilon = 0.3

    # load game and reset to initial state
    env = gym.make("Taxi-v3")
    env.reset()

    # initialize the Q table, states -> actions
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # train Q table
    Q_optimal = train_Qtable(Q, env, num_episodes, epsilon, gamma, learning_rate)

    # print(Q)

    launch_game(Q_optimal, env)

