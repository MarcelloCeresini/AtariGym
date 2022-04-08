import numpy as np
import tensorflow as tf

def calculate_eps(total_frames, replay_start_size, final_exploration_frame, inti_epsilon, end_epsilon):
    fraction_frames = (total_frames - replay_start_size) / final_exploration_frame
    return max( 
        (1-fraction_frames)*inti_epsilon + fraction_frames*end_epsilon, 
        end_epsilon
    )

def eps_greedy_policy(eps, env, model, p_obs):
    if np.random.uniform() < eps:
        return env.action_space.sample()
    else:
        return tf.cast(tf.math.reduce_max(model(tf.expand_dims(p_obs, axis=0))), dtype=tf.int32) # necessary to give batch=1


def process_gradient(g):
    return tf.math.maximum(1., tf.math.minimum(-1., -g))