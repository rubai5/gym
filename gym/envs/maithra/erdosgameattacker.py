# gym like version

# imports for creating game
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

logger = logging.getLogger(__name__)

# main game code

class ErdosGameAttackerEnv(gym.Env):
    
    def __init__(self, K, potential, unif_prob, 
                 geo_prob, diverse_prob, state_unif_prob, high_one_prob,
                geo_high, unif_high, random_def, train_attacker=True,
                geo_ps=[0.45, 0.5, 0.6, 0.7, 0.8], hash_states=None):
        
        self.K = K
        self.unif_prob = unif_prob
        self.geo_prob = geo_prob
        self.diverse_prob = diverse_prob
        self.state_unif_prob = state_unif_prob
        assert(self.state_unif_prob + self.diverse_prob + self.geo_prob + self.unif_prob == 1), \
               "State generation probabilities do not add to 1"
        self.train_attacker=train_attacker
        self.random_def = random_def
        
        self.high_one_prob = high_one_prob

        self.geo_high = geo_high
        self.unif_high = unif_high
        self.geo_ps = geo_ps
        self.hash = hash_states

        # adding an attribute called model_state to pass in values to
        self.model_state = []

        # build up idxs to deal with large powers of 2
        idxs = range(self.K + 1)

        self.random = random
        self.weights = np.power(2.0, [-(self.K - i) for i in idxs])
        self.attacker = 0
        self.defender = 0
        self.steps = 0
        self.potential = potential
    
        # gym specific stuff
        self.action_space = spaces.Discrete(self.K)
        self.observation_space = spaces.ErdosState(self.K, self.potential, self.weights, 
                                                  [self.unif_prob, self.geo_prob, self.diverse_prob, self.state_unif_prob],
                                                  self.high_one_prob, self.geo_high, self.unif_high,
                                                  self.geo_ps, self.train_attacker)
        self.viewer = None
        
        # only one state
        self.state = None
        
        # Maithra: commenting out seed for now
        #self._seed()

    def potential_fn(self, state):
        return np.sum(state*self.weights)

    def set_model_state(self, model_state):
        self.model_state = model_state
        
    def _step(self, action):
        """
        Updates state and returns new state, reward, done, and any info (as a dict)
        """
        # update state and determine if game ends
        # return empty log for now
        self.update_game_state(action)
        print("updates: reward, done and state", self.reward, self.done)
        print(self.state)
        
        return self.state, self.reward, self.done, {"steps": self.steps, "visited" : self.hash}
    
    
    def update_game_state(self, action):
        """
        Function updates game_state, using state  (which is concat of [A, B])
        """
        print("state is", self.state, self.potential_fn(self.state))
        # Create sets from action
        A = np.zeros(self.K+1).astype("int")
        B = np.zeros(self.K+1).astype("int")
        # making game easier
        nonzeros = np.where(self.state >0)[0]
        # if A has all nonzero terms
        if action+1 > np.max(nonzeros):
            # final nonzero included in B
            idx = np.max(nonzeros) - 1
        # if B has all nonzero terms
        elif action < np.min(nonzeros):
            # first nonzero included in A
            idx = np.min(nonzeros)
        else:
            idx = action

        A[:idx+1] = self.state[:idx+1]
        B[idx+1:] = self.state[idx+1:]
        print("action is", action)
        print("A and B are")
        print(A)
        print(B)
        # record game state
        if self.hash != None:
            try:
                self.hash[self.state] += 1
            except KeyError:
                pass

        # record
        self.steps += 1

        # decide which set is destroyed and which is pushed
        destroy, push = self.destroy_push(A, B)
        self.state -= (destroy + push)
        push = push[:-1]
        push = np.insert(push, 0, 0)
        self.state += push

        assert(np.all(self.state >= 0)), print("State negative!", self.state)
        
        if self.state[-1] > 0:
            self.attacker += 1
            #self.past_rewards.append(-1)
            self.reward = 1
            self.done = True

        if np.all(self.state == 0):
            self.defender += 1
            #self.past_rewards.append(1)
            self.reward = -1
            self.done = True
    
    def destroy_push(self, A, B):
        # picks sets to destroy and push
        random = np.random.binomial(1, self.random_def)
        if not random:
            potA = self.potential_fn(A)
            potB = self.potential_fn(B)
            if potA >= potB:
                return A, B
            else:
                return B, A
        else:
            randomized = np.random.binomial(1, 0.5)
            if randomized:
                return A, B
            else:
                return B, A   
    
    
    def _reset(self):
        """
        Note: when playing adversarially, call set_model_state
        before reset
        """
        self.state = self.observation_space.sample().astype("int")
        self.done = False
        self.reward = 0
        self.steps = 0
        
        return self.state

    def ground_truth(self):
        # returns index for which state >= 0.5*potential
        if np.sum(self.state) == 0:
            return np.array([self.K])
        weighted = self.state*self.weights
        for i in range(len(self.state)):
            if np.sum(weighted[:i])/np.sum(weighted) >= 0.5:
                break
        return np.array([i])

    def labels(self):
        idx = self.ground_truth()[0]
        set_labels = np.zeros(self.K+1)
        set_labels[idx] += 1
        return set_labels
