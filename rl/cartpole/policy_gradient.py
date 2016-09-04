
import gym
import numpy as np

def generate_episode(env, weight):
    episode = []
    pre_observation = env.reset()

    t = 0
    #generate 1 episodes for training.
    while 1:
        #env.render()
        pi, action = choose_action(weight, pre_observation)
    
        observation, reward, done, info = env.step(action)
        episode.append([pre_observation, action, pi, reward])
        pre_observation = observation
    
        t += 1
        if done or t > 1000:
            break
    return episode

def evaluate_given_parameter_sigmoid(env, weight):
    observation = env.reset()
    total_reward = 0.
    for t in range(1000):
        env.render()
        weighted_sum = np.dot(weight, observation)
        pi = 1 / (1 + np.exp(-weighted_sum))
        if pi > 0.5:
            action = 1
        else:
            action = 0
        
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def monte_carlo_policy_gradient(env):

    learning_rate = -0.0001
    best_reward = -100.0

    weight = np.random.rand(4)

    for iiter in xrange(1000):

        cur_episode = generate_episode(env, weight)
        for t in range(len(cur_episode)):
             
            observation, action, pi, reward = cur_episode[t]

            #update theta
            weight += learning_rate*(1-pi)*np.transpose(-observation)*reward

    cur_reward = evaluate_given_parameter_sigmoid(env, weight)
    print 'Monte-Carlo policy gradient get reward', cur_reward
 
def choose_action(weight, observation):

    weighted_sum = np.dot(weight, observation)
    pi = 1 / (1 + np.exp(-weighted_sum))
    if pi > 0.5:
        action = 1
    else:
        action = 0
    return pi, action

def actor_critic_policy_gradient(env):
    gamma = 1

    p_weight = np.random.rand(4)
        
    #weight for value function
    v_weight = np.random.rand(4)

    p_learning_rate = -0.0001
    v_learning_rate = -0.0001

    done = True

    for iiter in xrange(1000):

        t = 0
        while 1:
            if done:
                print 'start new training...'
                print 'p_weight', p_weight
                print 'v_weight', v_weight

                pre_observation = env.reset()
                pre_pi, pre_action = choose_action(p_weight, pre_observation)
        
                pre_phi = pre_observation
                pre_q = np.dot(v_weight, pre_phi)

            #env.render()

            observation, reward, done, info = env.step(pre_action)

            pi, action = choose_action(p_weight, observation)
            
            phi = observation
            q = np.dot(v_weight, phi)

            delta = reward + gamma*q - pre_q

            p_weight += p_learning_rate*(1-pre_pi)*np.transpose(-pre_observation)*pre_q

            v_weight += v_learning_rate*delta*np.transpose(pre_phi)

            pre_pi = pi
            pre_observation = observation
            pre_q = q
            pre_phi = phi
            pre_action = action

            t += 1
            if done:
                break

    cur_reward = evaluate_given_parameter_sigmoid(env, p_weight)
    print 'Actor critic policy gradient get reward', cur_reward

env = gym.make('CartPole-v0')

#env.monitor.start('cartpole-hill/', force=True)
actor_critic_policy_gradient(env)
#env.monitor.close()

monte_carlo_policy_gradient(env)
