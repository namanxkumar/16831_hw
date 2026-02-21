import numpy as np
import time
import copy

############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, *_ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        ac = policy.get_action(ob)
        ac = ac[0]
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, *_ = env.step(ac)
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        if done or steps >= max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    paths = []
    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
    return paths

############################################
# BONUS: Parallel trajectory collection
############################################

def _worker_collect(args):
    """Worker function for parallel trajectory collection."""
    env_name, policy_state, max_path_length, discrete, ob_dim, ac_dim, n_layers, size, nn_baseline = args
    import gym
    from rob831.policies.MLP_policy import MLPPolicyPG

    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_

    env = gym.make(env_name)
    # Create a fresh policy and load weights
    policy = MLPPolicyPG(ac_dim, ob_dim, n_layers, size, discrete=discrete, nn_baseline=nn_baseline)
    policy.load_state_dict(policy_state)
    policy.eval()

    path = sample_trajectory(env, policy, max_path_length)
    env.close()
    return path


_parallel_pool = None

def _get_pool(n_workers):
    global _parallel_pool
    if _parallel_pool is None:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        _parallel_pool = ctx.Pool(processes=n_workers)
    return _parallel_pool


def sample_trajectories_parallel(env, policy, min_timesteps_per_batch, max_path_length, n_workers=4):
    """Parallel version of sample_trajectories using multiprocessing."""
    env_name = env.spec.id
    policy_state = {k: v.cpu() for k, v in policy.state_dict().items()}
    worker_args = (env_name, policy_state, max_path_length,
                   policy.discrete, policy.ob_dim, policy.ac_dim,
                   policy.n_layers, policy.size, policy.nn_baseline)

    pool = _get_pool(n_workers)

    paths = []
    timesteps_this_batch = 0
    while timesteps_this_batch < min_timesteps_per_batch:
        results = pool.map(_worker_collect, [worker_args] * n_workers)
        for path in results:
            paths.append(path)
            timesteps_this_batch += get_pathlength(path)
            if timesteps_this_batch >= min_timesteps_per_batch:
                break

    return paths, timesteps_this_batch

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data
