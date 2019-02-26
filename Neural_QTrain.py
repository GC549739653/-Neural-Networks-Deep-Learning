import gym
import tensorflow as tf
import numpy as np
import random

tf.reset_default_graph()
# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
#EPISODE = 600  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.98# discount factor
INITIAL_EPSILON = 1.0 # starting value of epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
EPSILON_DECAY_STEPS = 20 # decay period
SYNC_PERIOD = 5   # network variable sync period
UPDATE_RATE=1 # target network weight update coefficient
MINIBATCH_SIZE = 124 # minibatch size in training
MAX_TRAIN_EPISODE = 100 #max train num
MIN_TRAIN_STEPS = 5000 #min train num
QUIT_REWARDS_THRESHOLD=196#
MAX_REWARD_NUM=11 # maximum mean reward num
MEMORY_SIZE=20000#memory size

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

#--------------------------------------------------------------------------------------------------------
#put value in a dict then let player use this value easily....
parameter = dict()
parameter['memory_size'] = MEMORY_SIZE
parameter['input_dim'] = STATE_DIM
parameter['output_dim'] = ACTION_DIM
parameter['update_rate'] = UPDATE_RATE
parameter['state__in'] = state_in
parameter['target__in'] = target_in
parameter['action__in'] = action_in
parameter['max_reward_num'] = MAX_REWARD_NUM
parameter['minibatch_size'] = MINIBATCH_SIZE
parameter['synchronize_period'] = SYNC_PERIOD
parameter['gamma'] = GAMMA
parameter['max_training_episode'] = MAX_TRAIN_EPISODE
parameter['min_training_steps'] = MIN_TRAIN_STEPS
parameter['quit_rewards_threshold'] = QUIT_REWARDS_THRESHOLD

#--------------------------------------------------------------------------------------------------------
class DQN(object):
    # build a double q learning nerwords to help us do next step....
    def __init__(self, input_dim, output_dim, update_rate=1.0):
        self.n_input = input_dim
        self.n_output = output_dim
        # we use only one layer here
        self.n_layer1 = 64
        self.update_rate = update_rate
        
    def get_target_q_values(self):
        return tf.stop_gradient(self.target_net_qvalues)

    def get_eval_q_values(self):
        return self.eval_net_qvalues

    def get_loss(self):
        return self.loss

    def get_optimizer(self):
        return self.optimizer

    def create_net(self, state__in):
        #We build our network here.....with many tests,i found use 1 layer is better than 2 layers...
        layer1 = tf.layers.dense(inputs=state__in,
                             units=self.n_layer1,
                             activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(),
                             use_bias=True,
                             name='layer1')

        outputs = tf.layers.dense(inputs=layer1,
                                  activation=None,
                                  units=self.n_output,
                                  use_bias=True,
                                  kernel_initializer=tf.random_normal_initializer(),
                                  name='out')
        return outputs

    def create(self, state__in, target__in, action__in):
     #create internal dqn 
        self.eval_net_qvalues = tf.make_template('eval', self.create_net)(state__in)
        self.target_net_qvalues = tf.make_template('target', self.create_net)(state__in)


        # we get loss values here
        q_val_select = tf.reduce_sum(tf.multiply(self.eval_net_qvalues, action__in), axis=1)
        a=target__in - q_val_select
        self.loss = tf.reduce_mean(tf.square(a))

        # get optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)



    def getvariables(self, name_scope):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name_scope)
        return trainable_vars

    def Synchonize_network(self):
        vars_in_eval_qnet = self.getvariables('eval')
        vars_in_target_qnet = self.getvariables('target')
        L = []
        for (var_eval, var_target) in zip(vars_in_eval_qnet, vars_in_target_qnet):
            a = var_eval * self.update_rate + var_target * (1 - self.update_rate)
            L.append(var_target.assign(a, use_locking=True))
        return L

    def predict_q_value(self, state__in):
        q_values = self.target_net_qvalues.eval(feed_dict={state: state__in})
        # predict value and will not be used
        re = tf.stop_gradient(q_values)
        return re

#--------------------------------------------------------------------------------------------------------
class player(object):
    # player is who play this game, and each function is what to do ...
    def __init__(self, parameter):
        self.parameter = parameter
        self.replay_buffer = creatbuffer(parameter['memory_size'])
        self.double_q_nets = DQN(parameter['input_dim'], parameter['output_dim'], parameter['update_rate'])
        self.double_q_nets.create(parameter['state__in'], parameter['target__in'], parameter['action__in'])
        self.n_training_steps = 0
        self.max_reward_num = parameter['max_reward_num']
        self.latest_rewards = []
        self.train_stop_flag = False
        self.synchronize = self.double_q_nets.Synchonize_network()

    def is_train_stop(self, current_episode):
        # over max episode
        if current_episode > self.parameter['max_training_episode']:
            return True
        # At least the number of training required
        if self.n_training_steps < self.parameter['min_training_steps']:
            return False
        # if latest_reward is not bigger than max_reward_num
        if len(self.latest_rewards) < self.max_reward_num:
            return False

        mean_rewards = np.mean(self.latest_rewards)
        if mean_rewards >= self.parameter['quit_rewards_threshold']:
            return True
        else:
            return False

    def caculate_target(self, minibatch__size):
        state = [elem[0] for elem in minibatch__size]
        next_state = [elem[2] for elem in minibatch__size]
        action = [elem[3] for elem in minibatch__size]
        ns_qvalues = self.double_q_nets.target_net_qvalues.eval(feed_dict={state_in: next_state})
        # this step is to calculate target value each state.
        target = []

        for i in range(len(minibatch__size)):
            target_reward = minibatch__size[i][1]
            #print( target_reward)
            done = minibatch__size[i][4]
            #print(done)
            if not done:
                ns_max_qvalue = np.max(ns_qvalues[i])
                target_reward += self.parameter['gamma'] * ns_max_qvalue
            target.append(target_reward)
        re = dict({self.parameter['target__in']: target, self.parameter['action__in']: action, self.parameter['state__in']: state})
        return re

    def remember(self, experience):
        # record things into buffer
        self.replay_buffer.record(experience)

    def sync_networks(self):
        # check  synchronize parameters in target net and evaluation net
        if self.n_training_steps != 0 and self.n_training_steps % self.parameter['synchronize_period'] == 0:
            session.run([self.synchronize])
        return

    def record_rewards(self, episode_reward):
        self.latest_rewards.append(episode_reward)
        if len(self.latest_rewards) >= self.max_reward_num:
            self.latest_rewards.pop(0)
        return

    def train(self, current_episode):
        if self.train_stop_flag == True:
            return

        if self.is_train_stop(current_episode) == True:
            self.train_stop_flag = True
            return

        self.sync_networks()
        minibatch__size = self.replay_buffer.next_minibatch(self.parameter['minibatch_size'])
        # do not have enough experiences
        if None == minibatch__size or len(minibatch__size) == 0:
            return
        else:
            pass
        feed_data = self.caculate_target(minibatch__size)
        session.run([self.double_q_nets.get_optimizer()], feed_dict=feed_data)

        # start another training steps
        self.n_training_steps += 1

    def predict_q_value(self):
        re = self.double_q_nets.predict_q_value(self.parameter['state__in'])
        return re

#--------------------------------------------------------------------------------------------------------
class creatbuffer(object):
# creatbuffer to record experiences and go on replay........
    def __init__(self, memory_size=10000):
        self.cursor = 0
        self.capacity = memory_size
        self.memory = list()

    def record(self, exp):
        #write exp
        if len(self.memory) < self.capacity:
            self.memory.append(exp)
        else:
            # overwrite exp
            self.memory[self.cursor] = exp
            self.cursor = (self.cursor + 1) % self.capacity

    def size(self):
        #get size of memory
        relen = len(self.memory)
        return relen

    def next_minibatch(self, minibatch_size=100):
        n_size = np.minimum(len(self.memory), minibatch_size)
        if n_size != 0:
            pass
        else:
            return []
        return random.sample(self.memory, n_size)



def get_target_value(agent, state, reward, next_state, action, done, episode_reward):
    agent.remember([state, reward, next_state, action, done])
    episode_reward += reward
    state = next_state
    agent.train(episode)
#--------------------------------------------------------------------------------------------------------
# TODO: Define Network Graph
agent = player(parameter)
# TODO: Network outputs
q_values = agent.double_q_nets.get_eval_q_values()
q_action = tf.argmax(q_values)

# TODO: Loss/Optimizer Definition
loss = agent.double_q_nets.get_loss()
optimizer = agent.double_q_nets.get_optimizer()

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

#--------------------------------------------------------------------------------------------------------
# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


#--------------------------------------------------------------------------------------------------------
# Main learning loop
env.seed(0)
random.seed(0)
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    if epsilon > FINAL_EPSILON:
        epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    episode_reward = 0
    for step in range(STEP):
        if agent.is_train_stop(episode) == True:
            break
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
       # nextstate_q_values = q_values.eval(feed_dict={
        #    state_in: [next_state]
        #})

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        #get_target_value(agent, state, reward, next_state, action, done, episode_reward)
        agent.remember([state, reward, next_state, action, done])
        episode_reward += reward
        state = next_state
        agent.train(episode)
        if done:
            break
    agent.record_rewards(episode_reward)
#--------------------------------------------------------------------------------------------------
    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()

