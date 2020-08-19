import gym
import minerl
import logging
import random
import matplotlib.pyplot as plt

def plot_stats(angles, rewards):
    plt.subplot(211)
    plt.plot(angles)
    plt.subplot(212)
    plt.plot(rewards)
    plt.show()

#logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLNavigateDense-v0')

obs = env.reset()
done = False
net_reward = 0
rewards = []
angles = []

while not done:
    action = env.action_space.noop()

    noise = 0
    if random.random() < 0.05:
        noise = (random.random()*2-1)*180
    action['camera'] = [0, 0.05*obs["compassAngle"]+noise]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    rewards.append(net_reward)
    angles.append(obs['compassAngle'])
    print("Total reward: ", net_reward)

plot_stats(angles, rewards)
