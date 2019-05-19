import numpy as np
from physics_sim import PhysicsSim

import math
import gym
import gym.spaces
import gym.envs

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.init_pose = init_pose if init_pose is not None else np.array([0., 0., 10.])
        
        self.viewer = None
      
    def get_noise(self):
        return abs(np.random.normal(.1, .5, 1)[0])

    def get_reward(self, time_elaps, time_limit, start_position, end_position, target_position, done):
        """Uses current pose of sim to return reward."""
        z_axis_index = 2
        reward = 0
        
        # the difference between the final and start position
        delta_pose = end_position - start_position

        # if the quadcopter crashes
        if done and time_elaps < time_limit:
            reward = -50

        # if the quadcopter finishes above the start position
        if end_position > start_position:
            reward += delta_pose

        # if the quadcopter finishes above the target position
        if end_position > target_position:
            reward += 10

        # if don't fall, the reward cannot be negative
        if reward < 0 and end_position > 0:
            reward += .5 * abs(self.get_noise()) + 1

        return .7 + np.tanh(reward) * .3 

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(self.sim.time, self.sim.runtime, self.init_pose[2], self.sim.pose[2], self.target_pos[2], done)  # self.get_rewards() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        # reward = reward * (.01 if self.sim.pose[2] < 1 else self.sim.pose[2])
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
