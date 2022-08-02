import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        self.hidden_sz = 300
        self.dense1 = tf.keras.layers.Dense(self.hidden_sz)
        self.dense2 = tf.keras.layers.Dense(self.num_actions, activation = 'softmax')

        self.dense3 = tf.keras.layers.Dense(self.hidden_sz)
        self.dense4 = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def call(self, states):

        pass1 = self.dense1(states)
        probs = self.dense2(pass1)
        return probs


    def value_function(self, states):

        pass1 = self.dense3(states)
        val = self.dense4(pass1)
        return val


    def loss(self, states, actions, discounted_rewards):

        pdf_action = self.call(states)
        val = self.value_function(states)
        val=tf.squeeze(val)
        advs = discounted_rewards - val
        action = []


        for i, act in enumerate(actions):
            action.append((i, act))


        gathered = tf.gather_nd(pdf_action, action)
       
        log=tf.math.log(gathered)
        neg_log=-1 * log 

        
        
        adv_log = neg_log * tf.stop_gradient(advs)
        actor = tf.reduce_sum(adv_log)
        critic = tf.reduce_sum(advs**2)
        
        tloss = actor + critic
        return tloss