import numpy as np
"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return np.random.randint(0,9)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass

class EpsilonGreedy:

    def __init__(self):
        # self.average is a (n, p) matrix where x=10 and p=2 
        # for each x is the arm x is in range(10):[0,..,9]
            # average[x]:
                 1.    average reward for arm x 
                # 2.    count of reward used to compute the average reward for arm x
        self.average = np.zeros((10,2))

        # step_number is the step of iteration
        self.step_number = 0

        # epsilon is the probability of picking RANDOMLY an arm
        self.epsilon = 1

    def act(self, observation):
        # Based on Epsilon value, we return the arm that we pick for this step
        if np.random.random(1)[0] < self.epsilon:
            return np.random.randint(0,9)
        else:
            return self.get_arm_with_the_highest_average_reward()

    def reward(self, observation, action, reward):
        self.step()
        self.update_average(action, reward)
        print(self.average)

    """
    METHODS:
        - get_arm_with_the_highest_average_reward:
            Pick the arm with the CURRENT highest average reward
        - update average:    
            Update the value of the matrix of average reward step by step

    """
    def get_arm_with_the_highest_average_reward(self):
        # List of rewards
        rewards = list(self.average[:,0])
        arm = rewards.index(max(rewards))
        print('The arm with the highest average reward is: \t', arm)
        print('Type of arm is: ', type(arm))
        return arm

    def update_average(self, action, reward):
        self.average[action][0] = \
                (self.average[action][0] * self.average[action][1] + reward) \
                / (self.average[action][1] + 1)
        self.average[action][1] += 1

    def step(self):
        self.step_number += 1
        self.epsilon *= 0.9
        print('Step\t' + str(self.step_number))


class UCBAgent:

    def __init__(self):
        # self.average is a (n, p) matrix where x=10 and p=2 
        # for each x is the arm x is in range(10):[0,..,9]
            # average[x]:
                # 1.    average reward for arm x 
                # 2.    count of arm x usage
                # 3.    Upper Bound (mathematical formula) 
        self.data = np.zeros((10,3))

        # step_number is the step of iteration
        self.step_number = 0

        # trigger to start using UCB algorithm
        self.step_trigger = 80

    def act(self, observation):
        # We first play each arm one time
        if self.step_number < self.step_trigger:
            return self.step_number % 10
        # After playing each arm once
        else:
            return self.select_arm()

    def reward(self, observation, action, reward):
        self.step()
        self.update_data(action, reward)
        print(self.data)

    """
    METHODS:
        - select_arm:
            return arm with the highest upper bound value 
        - update data:    
            update the value of data (average reward, count of use, upper bound)
            THIS CODE IS WRITTEN BY YASSINE BELMAMOUN (IN CASE OF COPY/PASTE)
    """
    def select_arm(self):
        ucbs = list(self.data[:,2])
        arm = ucbs.index(max(ucbs))
        return arm

    def update_data(self, action, reward):
        # Upper bound computation (Start after initialisation of the average)
        self.data[action][2] = self.data[action][0] + \
            np.sqrt(\
                      2 * np.log(self.step_number + 1) / \
                      self.data[action][1]\
                     )

        # Count arm usage
        self.data[action][1] += 1

        # Average computation
        self.data[action][0] = \
                (self.data[action][0] * self.data[action][1] + reward) \
                / (self.data[action][1] + 1)


    def step(self):
        self.step_number += 1
        print('------------ Step\t' + str(self.step_number))


# Choose which Agent is run for scoring
Agent = UCBAgent 

