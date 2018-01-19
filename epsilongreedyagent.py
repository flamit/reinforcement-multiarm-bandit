import numpy as np

class EpsilonGreedy:

    def __init__(self):
        # self.average is a (n, p) matrix where x=10 and p=2 
        # for each x is the arm x is in range(10):[0,..,9]
            # average[x]:
                # 1.    average reward for arm x 
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