import numpy as np
"""
Contains the definition of the agent that will run in an
environment.
"""

class MyAgent:

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
        self.step_trigger = 20

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


# Choose which Agent is run for scoring
Agent = MyAgent 

