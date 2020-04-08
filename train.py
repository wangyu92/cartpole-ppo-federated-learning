import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import copy
import gym
import datetime

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from actor_critic import ActorCritic

#############################################################################
# Process                                                                   #
#############################################################################
# - There is a fixed set of K clients, each with a fixed local dataset.
# [Start round]
# 1. A random fraction of C of clients is selected, and the server sends the current global model.
# 2. Each selected client then performs local computation based on the global state and its dataset.
# 3. Then each client sends an update to the server. 

#############################################################################
# Hyperparameters                                                           #
#############################################################################
NUM_CLIENTS     = 100
FRACTION        = 0.1

T_HORIZON       = 32
N_MINIBATCH     = 1
LEARNING_RATE   = 5e-4
EPS             = 0.2
N_EPOCH         = 8

GAMMA           = 0.99
LAMBDA          = 0.95

# CUDA
CUDA            = False
CUDA_INDEX      = 2

#############################################################################
# Util functions                                                            #
#############################################################################

def cuda_device(enable=True, idx=0):
    if enable:
        d_str = 'cuda:' + str(idx)
        device = torch.device(d_str if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device

#############################################################################
# Central                                                                   #
#############################################################################
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def test_model(model):
    model.eval()
    env = gym.make('CartPole-v1')

    score = []

    for _ in range(20):
        score.append(0.)
        s = env.reset()
        while True:
            a = model.action(s)
            s, r, d, _ = env.step(a)
            score[-1] += r
            if d:
                break

    return np.mean(score)


def central(model_queues, update_queues):
    device = cuda_device(CUDA, CUDA_INDEX)

    # Create model
    env = gym.make('CartPole-v1')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, lr=LEARNING_RATE)
    model.to(device)
    print(test_model(model))
    print()

    # tensorboard
    s_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = '../tensorboard/' + 'CartPole-PPO_FED_' + s_time
    train_summary_writer = SummaryWriter(dir_path)

    num_rounds = 0
    while True:
        print('# of Round : ', num_rounds)

        #############################################################################
        # Select random fraction of clients, and global model to each client        #
        #############################################################################
        model.train()

        num_cli_sel = max(int(FRACTION * NUM_CLIENTS), 1)
        clients_sel = np.random.choice(range(NUM_CLIENTS), num_cli_sel, replace=False)
        print('selected clients : ', clients_sel)

        for idx in clients_sel:
            model_state = model.state_dict()
            model_queues[idx].put((model_state))

        #############################################################################
        # Receieve updates from clients                                             #
        #############################################################################
        local_states = []
        for idx in clients_sel:
            lstate, losses, losses_p, losses_v, entropies = update_queues[idx].get()
            local_states.append(lstate)

        #############################################################################
        # Train                                                                     #
        #############################################################################
        averaged_states = average_weights(local_states)
        model.load_state_dict(averaged_states)

        #############################################################################
        # Test                                                                      #
        #############################################################################
        score = test_model(model)
        print(score)

        train_summary_writer.add_scalar('Loss/Total', np.mean(losses), num_rounds)
        train_summary_writer.add_scalar('Loss/Actor', np.mean(losses_p), num_rounds)
        train_summary_writer.add_scalar('Loss/Critic', np.mean(losses_v), num_rounds)
        train_summary_writer.add_scalar('Loss/Entropy', np.mean(entropies), num_rounds)
        train_summary_writer.add_scalar('test_epi_rewards', score, num_rounds)
        train_summary_writer.flush()

        num_rounds += 1
        print()

#############################################################################
# Local                                                                     #
#############################################################################
def compute_gae(rewards, values, values_n, dones):
    td_target = rewards + GAMMA * values_n * (1 - dones)
    delta = td_target - values
    gae = np.append(np.zeros_like(rewards), [0], axis=-1)
    for i in reversed(range(len(rewards))):
        gae[i] = GAMMA * LAMBDA * gae[i + 1] * (1 - dones[i]) + delta[i]
    gae = gae[:-1]
    return gae, td_target


def train_step(model, states, actions, probs, rewards, dones, statesn):
    device = model.getdevice()

    # compute return and advantages
    values = model.values(states)
    valuesn = model.values(statesn)
    advs, returns = compute_gae(rewards, values, valuesn, dones)
    
    # advs = returns - values

    # # Normalize the advantages
    # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    # convert from numpy to tensors
    states = torch.tensor(states.copy(), device=device).float()
    actions = torch.tensor(actions.copy(), device=device).long()
    probs = torch.tensor(probs.copy(), device=device).float()
    advs = torch.tensor(advs.copy(), device=device).float()
    returns = torch.tensor(returns.copy(), device=device).float()
    values = torch.tensor(values.copy(), device=device).float()

    # reshape the data
    actions = actions[:, None]
    probs = probs[:, None]
    advs = advs[:, None]
    returns = returns[:, None]
    values = values[:, None]

    # compute gradient and updates
    optimizer = model.optimizer
    optimizer.zero_grad()
    ppreds, vpreds = model(states)
    loss_policy = loss_policy_fn(ppreds, actions, probs, advs)
    loss_value = loss_value_fn(vpreds, returns, values)
    vf_coef = 0.5

    loss = (loss_policy) + (vf_coef * loss_value)
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy(), loss_policy.detach().cpu().numpy(), loss_value.detach().cpu().numpy()


def loss_policy_fn(preds, actions, oldprobs, advs):
    """
    preds = (batch_size, action_size)
    actions = (batch_size, 1)
    oldprobs = (batch_size, 1)
    advs = (batch_size, 1)
    """
    probs = torch.gather(preds, dim=-1, index=actions)
    ratio = torch.exp(torch.log(probs) - torch.log(oldprobs))
    surr1 = ratio * advs
    surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * advs

    loss = -torch.min(surr1, surr2)
    return loss.mean()


def loss_value_fn(preds, returns, oldvpred):
    vpredclipped = oldvpred + torch.clamp(preds - oldvpred, -EPS, EPS)
    vf_losses1 = (preds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2

    loss = .5 * torch.max(vf_losses1, vf_losses2).mean()
    return loss.mean()


def local(i, model_queue, update_queue):
    device = cuda_device(CUDA, CUDA_INDEX)

    env = gym.make('CartPole-v1')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, lr=LEARNING_RATE)
    model.to(device)

    states = np.empty((T_HORIZON, env.observation_space.shape[0]))
    statesn = np.empty((T_HORIZON, env.observation_space.shape[0]))
    probs = np.empty((T_HORIZON,))
    actions = np.empty((T_HORIZON,), dtype=np.int32)
    rewards = np.empty((T_HORIZON,))
    dones = np.empty((T_HORIZON,))

    entropies = np.empty((T_HORIZON,))

    sn = env.reset()
    while True:
        #############################################################################
        # Receieve global model                                                     #
        #############################################################################
        global_state = model_queue.get()
        model.load_state_dict(global_state)

        #############################################################################
        # Experience                                                                #
        #############################################################################
        for t in range(T_HORIZON):
            states[t] = sn.copy()
            actions[t], probs[t], entropies[t] = model.action_sample(states[t])
            sn, rewards[t], dones[t], _ = env.step(actions[t])
            statesn[t] = sn.copy()

            if dones[t]:
                sn = env.reset()
                statesn[t] = sn.copy()

        #############################################################################
        # Local Computation (train)                                                 #
        #############################################################################
        losses = []
        losses_p = []
        losses_v = []
        for _ in range(N_EPOCH): 
            l, p, v = train_step(model, states, actions, probs, rewards, dones, statesn)
            losses.append(l)
            losses_p.append(p)
            losses_v.append(v)

        #############################################################################
        # Send model to the server                                                  #
        #############################################################################
        local_state = copy.deepcopy(model.state_dict())
        update_queue.put((local_state, losses, losses_p, losses_v, entropies))


if __name__  == '__main__':
    ############################################################################
    # Create queues for communication                                          #
    ############################################################################
    model_queues = []
    update_queues = []
    for i in range(NUM_CLIENTS):
        model_queues.append(mp.Queue(1))
        update_queues.append(mp.Queue(1))

    ############################################################################
    # Create processes                                                         #
    ############################################################################
    coordinator = mp.Process(target=central, args=(model_queues, update_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_CLIENTS):
        agents.append(mp.Process(target=local, args=(i, model_queues[i], update_queues[i])))

    for i in range(NUM_CLIENTS):
        agents[i].start()