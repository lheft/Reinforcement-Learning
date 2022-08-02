import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline

def visualize_episode(env, model):

    done = False
    state = env.reset()
    env.render()

    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        prob = model.call(newState)
        newProb = np.reshape(prob, prob.shape[1])
        action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)

        state, _, done, _ = env.step(action)
        env.render()

def visualize_data(total_rewards):


    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):

    reward_list=[]
    d = 0
    for num in range(len(rewards)):
        gi = (discount_factor*num)
        d = d+ ( gi * rewards[num])
        reward_list.append(d)
    rewards=reward_list[::-1]
    return rewards


def generate_trajectory(env, model):

    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        shape=[1,-1]
        reshape=tf.reshape(state,shape)
        
        pdf = model.call(reshape)
        squee = np.squeeze(pdf)
        
        range=[0,1]
        action = np.random.choice(range, p=squee)
        
        states.append(state)
        actions.append(action)

        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)

    return states, actions, rewards


def train(env, model):


    with tf.GradientTape() as tape:
        stat, act, rwd = generate_trajectory(env, model)
        dr = discount(rwd)
        s_tensor=tf.convert_to_tensor(stat)
        a_tensor=tf.convert_to_tensor(act)
        dr_tensor=tf.convert_to_tensor(dr)
        loss=model.loss(s_tensor,a_tensor,dr_tensor)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    trwd = np.array(rwd)
    toe = tf.reduce_sum(trwd)
    return toe

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("CartPole-v1") # environment
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions) 
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    numpisodes = 650

    tr_reward= []
    for _ in range(numpisodes):
        train_r = train(env, model)
        tr_reward.append(train_r)

    last50 = tr_reward[-50:]
    print(tf.reduce_sum(last50)/50)
    visualize_data(tr_reward)


if __name__ == '__main__':
    main()