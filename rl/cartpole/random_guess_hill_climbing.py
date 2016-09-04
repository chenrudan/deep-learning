
import gym
import numpy as np

def evaluate_given_parameter_by_sign(env, weight):
    observation = env.reset()
    total_reward = 0.
    for t in range(1000):
        env.render()
        weighted_sum = np.dot(weight, observation)
        if weighted_sum >= 0:
            action = 1
        else:
            action = 0
        
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def random_guess():
    env = gym.make('CartPole-v0')
    np.random.seed(10)
    best_reward = -100.0

    for iiter in xrange(1000):
        weight = np.random.rand(4)

        cur_reward = evaluate_given_parameter_by_sign(env, weight)
        if cur_reward > best_reward:
            best_reward = cur_reward
            best_weight = weight

        if best_reward == 1000:
            break
        
    print("Random guess algorithm best reward", best_reward)
    print("Random guess algorithm best weight", best_weight)

def hill_climbing():
    env = gym.make('CartPole-v0')
    best_reward = -100.0
    np.random.seed(10)
    best_weight = np.random.rand(4)

    for iiter in xrange(1000):
        weight = best_weight + np.random.normal(0, 0.01, 4)

        cur_reward = evaluate_given_parameter_by_sign(env, weight)
        if cur_reward > best_reward:
            best_reward = cur_reward
            best_weight = weight

        if best_reward == 1000:
            break
        
    print("Hill climbing algorithm best reward", best_reward)
    print("Hill climbing algorithm best weight", best_weight)

random_guess()
hill_climbing()
