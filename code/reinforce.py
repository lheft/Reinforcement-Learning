import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        
        # TODO: Define network parameters and optimizer
        self.hidden_sz = 400
        self.dense1 = tf.keras.layers.Dense(self.hidden_sz)
        act='softmax'
        self.dense2 = tf.keras.layers.Dense(self.num_actions, activation = act)
        
        lr=.001

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
        
    @tf.function
    def call(self, states):
       
        pass1 = self.dense1(states)
        probs = self.dense2(pass1)
        return probs

    def loss(self, states, actions, discounted_rewards):
       
        pdf_a = self.call(states)
        action = []
        loss = 0

        for i, act in enumerate(actions):
            action.append((i, act))
        
        gather = tf.gather_nd(pdf_a, action)

        log = tf.math.log(gather)
        nlog=log*-1
        z = nlog * discounted_rewards

        for _, lossi in enumerate(z):
            loss =loss + lossi
        return loss
