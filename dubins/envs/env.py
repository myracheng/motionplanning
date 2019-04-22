import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

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
        self.max_u = 10
        self.dt = 0.2
        self.arrived = False;
        low = np.array([0, 0, -np.pi])
        high = np.array([self.width, self.height, np.pi])
        self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        y_new = np.clip(y_new, 0, self.width)
        
        theta_new = theta + (u * dt) #how to update this based on u?
        if theta_new < -np.pi:
            theta_new = theta_new + (2 * np.pi)
        elif theta_new > np.pi:
            theta_new = theta_new - (2 * np.pi)
       
        distance = np.sqrt((x_new - self.x_g)**2 + (y_new - self.y_g)**2)
        if (distance < 5):
            reward = 100
            self.arrived = True
        else:
            reward = -distance

        self.state = np.array([x_new, y_new, theta_new])
        return self._get_obs(), reward, self.arrived, {}

    def reset(self):
        self.arrived = False;
        low = np.array([0, 0, -np.pi])
        high = np.array([self.width, self.height, np.pi])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        x, y, theta = self.state
        return np.array([x, y, theta])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)