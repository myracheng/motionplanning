import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class DubinsEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.height = 50  # grid height
        self.width = 50  # grid width
        self.x_g = 27 #goal x
        self.y_g = 13 #goal y
        self.v = 1 #fixed forward velocity
        self.max_u = 3
        self.dt = 0.2
        self.arrived = False;
        low = np.array([0, 0, -np.pi])
        high = np.array([self.width, self.height, np.pi])
        self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.obstacleList = [
        (36, 43, 2),
        (12, 16, 2),
        (24, 34, 2)
        ] 
        self.seed()

        self.fig, self.ax = plt.subplots(1)
        plt.ion()
        plt.show()
        self.ax.add_patch(patches.Circle((self.x_g, self.y_g), 5.0, facecolor='g'))
        for i in range(len(self.obstacleList)):
            self.ax.add_patch(patches.Circle((self.obstacleList[i][0], self.obstacleList[i][1]), self.obstacleList[i][2], facecolor='r'))
        self.state_circle = patches.Circle((0.,0.), 0.5, facecolor='b')
        self.ax.add_patch(self.state_circle)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def isobstacle(self, ix, iy):
        for (ox, oy, size) in self.obstacleList:
                dx = ox - ix
                dy = oy - iy
                d = dx * dx + dy * dy
                if d <= (size + 0.1) ** 2:
                    # print('hit obstacle')
                    return True # collision

        return False  # safe

    def step(self, u):
        if self.arrived:
            raise RuntimeError("Episode is done")

        x, y, theta = self.state

        dt = self.dt
        u = np.clip(u, -self.max_u, self.max_u)
        self.last_u = u # for rendering

        x_new = x + (self.v * (np.cos(theta))*dt)
        x_new = np.clip(x_new, 0, self.width)

        y_new = y + (self.v * (np.sin(theta))*dt)
        y_new = np.clip(y_new, 0, self.height)
        
        theta_new = theta + (u * dt)
        if theta_new < -np.pi:
            theta_new = theta_new + (2 * np.pi)
        elif theta_new > np.pi:
            theta_new = theta_new - (2 * np.pi)
       
        distance = np.sqrt((x_new - self.x_g)**2 + (y_new - self.y_g)**2)
        if self.isobstacle(x_new, y_new):
            reward = -50
        elif (distance < 5):
            reward = 200
            self.arrived = True
        else:
            reward = -distance

        self.state = np.array([x_new, y_new, theta_new])
        return self._get_obs(), reward, self.arrived, {}

    
    def reset(self):
        self.arrived = False
        low = np.array([0, 0, -np.pi])
        high = np.array([self.width, self.height, np.pi])
        self.state = self.np_random.uniform(low=low, high=high)
        while self.isobstacle(self.state[0], self.state[1]):
            # print(self.state[2])
            print("found that this is obstacle")
            self.state = self.np_random.uniform(low=low, high=high) #make sure we're not starting at an obstacle.
            # print(self.state[2])
            print("reset to" + str(self.state[0]) + "and " + str(self.state[1]))
        self.last_u = None
        return self._get_obs()

    def get_reward(self, x_new, y_new):
        if self.isobstacle(x_new, y_new):
            return -50
        distance = np.sqrt((x_new - self.x_g)**2 + (y_new - self.y_g)**2)
        if (distance < 5):
            return 100
        else:
            return -distance

    def _get_obs(self):
        x, y, theta = self.state
        return np.array([x, y, theta])

    def render(self):
        x, y, _ = self.state
        self.state_circle.remove()
        self.state_circle = patches.Circle((x,y), 0.5, facecolor='b')
        self.ax.add_patch(self.state_circle)
        
        '''
        for i in range(len(self.obstacleList)):
            self.ax.add_patch(patches.Circle((self.obstacleList[i][0], self.obstacleList[i][1]), self.obstacleList[i][2], facecolor='r')) 
        self.ax.add_patch(patches.Circle((self.x_g, self.y_g), 5, facecolor='g'))
        self.ax.add_patch(patches.Circle((x, y), 0.5, facecolor='b'))
        '''
        plt.ylim((0,50))
        plt.xlim((0,50))
        plt.draw()
        plt.pause(0.001)
        #plt.show()
        #return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
