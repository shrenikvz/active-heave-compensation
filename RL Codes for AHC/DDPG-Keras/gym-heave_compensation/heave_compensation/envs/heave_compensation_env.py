                                            
                                    ## Importing Packages ##

import gym
import numpy as np
from scipy.integrate import solve_ivp
from gym import spaces
from gym.utils import seeding
#import random

import logging
logger = logging.getLogger(__name__)

class HeaveEnv(gym.Env):

    def __init__(self, g= 9.81 ,k_oil = 1.8*10**9, Vc = 2*10**(-3), Dp = 40*10**(-6), Dm = 4*10**(-6), w_p = 45, Tp = 1, k = 200, r = 0.5, eta_m = 0.65, J_w = 150, d = 10**4, m = 10**3):
        
        
        # Parameters for State Space Model of the Winch
        
        self.max_control_input= 10              # Max Control Input
        self.g = g
        self.k_oil = k_oil
        self.Vc = Vc
        self.Dp = Dp
        self.Dm = Dm
        self.w_p = w_p
        self.Tp = Tp
        self.k = k
        self.r = r
        self.eta_m = eta_m 
        self.J_w = J_w
        self.d = d
        self.m = m
        self.viewer = None
        
        # Defining Action Space
        
        self.action_space = spaces.Box(
            low=-self.max_control_input,
            high=self.max_control_input, shape=(1,),
            dtype=np.int32
        )
       
        
        # Defining Observation Space
        
        high = np.array([1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(
            low= -high,
            high= high,
            dtype=np.float32
        )

        self.seed()
            

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, up,l,reference,reference_dot):
                
        
        zw_dot, ref_dot, zw, ref= self.state
        x_s, delta_p = self.state_1

                    ## Parameters for the State Space Model of the Winch ##
        
        m = self.m
        k_oil = self.k_oil
        Vc = self.Vc
        Dp = self.Dp
        Dm = self.Dm
        w_p = self.w_p
        Tp = self.Tp
        k = self.k
        r = self.r
        eta_m = self.eta_m
        J_w = self.J_w
        d = self.d
        
                                    ## System Matrix  ##
        
        A_11 = -1/Tp
        A_21 = -2*k_oil*Dp*w_p/Vc
        A_23 = 2*(k_oil/Vc)*Dm*(k/r)
        A_32 = -(r/(J_w + m*r**2))*Dm*k*eta_m
        A_33 = -d/(J_w + m*r**2)
        A_43 = 1
        
                                    ## Input Matrix ##
        
        B_11 = 1/Tp
        
                        ## Solving the LTI-SS Model for the Winch ##
        
        x_initial = [x_s, delta_p, zw_dot, zw]
        tspan = (0, 0.1)                        # Interval for Integration
        
        up = np.clip(up, -self.max_control_input, self.max_control_input)[0]
        self.last_up = up
                
        def winch_model(t, y):
            dy_dt = [A_11*y[0] + B_11*up, A_21*y[0] + A_23*y[2], A_32*y[1] + A_33*y[2], A_43*y[2]]
            return dy_dt
    
        x_solution = solve_ivp(lambda t, y: winch_model(t, y), [tspan[0], tspan[-1]], x_initial, t_eval = tspan)
        x_new = [0] * 4
        x_new[0] = x_solution.y[0][-1]
        x_new[1] = x_solution.y[1][-1]
        x_new[2] = x_solution.y[2][-1]
        x_new[3] = x_solution.y[3][-1]
        
        x_swash = np.clip(x_new[0], -1, 1)          # For cliping swash angle between -1 and 1
        
        zw_error = abs(x_new[3]+reference[l])
        zw_dot_error = abs(x_new[2] + reference_dot [l])
        
               
                                    ## Reward ##
        
        
        if zw_error <= 0.05:
            rewards = 1 - 20*zw_error - 1*zw_dot_error
        if zw_error > 0.05:
            rewards = -10*zw_error - 2*zw_dot_error
            
        
        
        self.state = np.array([x_new[2], reference_dot[l], x_new[3], reference[l]])   # Reeled Velocity, Reference_dot, Reeled Position, and Reference
        self.state_1 = np.array([x_swash, x_new[1]])                                  # Swash angle and Change in Pressure
        return self.state, rewards, False, {}

    def reset(self):
        high = np.array([0, 0, 0, 0])
        self.state = self.np_random.uniform(low=-high, high=high)
        high_1 = np.array([0, 0])
        self.state_1 = self.np_random.uniform(low=-high_1, high=high_1)
        self.last_up = None
        return self.state,self.state_1

    '''
    def _get_obs(self,l):
        t   = np.arange(0, 105, 0.1);
        reference   = 0.06*np.sin(0.2*t)
        zw_dot, zw, reference[l-1] = self.state
        return np.array([zw_dot, zw, reference[l-1]])
    
    '''
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
                                                ## END ##

