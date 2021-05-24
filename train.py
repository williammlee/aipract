"""
AI Pract Final Project
"""
import argparse
import os
from random import random, randint, sample
import pickle
import numpy as np
import torch
import torch.nn as nn

from src.Agent import Agent
from src.env import ChromeDino


def helper_arg():
    parameters = argparse.ArgumentParser(
        """Implementation of Q Learning to play Chrome Dino""")
    parameters.add_argument("--batch_size", type=int, default=16)
    parameters.add_argument("--optimizer", type=str,
                     choices=["sgd", "adam"], default="adam")
    parameters.add_argument("--lr", type=float, default=1e-4)
    parameters.add_argument("--gamma", type=float, default=0.99)
    parameters.add_argument("--initial_epsilon", type=float, default=0.1)
    parameters.add_argument("--final_epsilon", type=float, default=1e-4)
    parameters.add_argument("--num_decay_iters", type=float, default=200000)
    parameters.add_argument("--num_iters", type=int, default=200000)
    parameters.add_argument("--replay_memory_size", type=int, default=50000)
    parameters.add_argument("--saved_folder", type=str, default="trained_models")

    resulting_arg = parameters.parse_args()
    return resulting_arg


def train_func(input):
    
    torch.manual_seed(123)
    model = Agent()
    optimizer = torch.optim.Adam(model.parameters(), lr=input.lr)
    if not os.path.isdir(input.saved_folder):
        os.makedirs(input.saved_folder)
    check_path = os.path.join(input.saved_folder, "chrome_dino.pth")
    mem_path = os.path.join(input.saved_folder, "replay_memory.pkl")

    if not os.path.isfile(check_path):
        iter = 0
    else:
        check = torch.load(check_path, map_location='cpu')
        iter = check["iter"] + 1
        model.load_state_dict(check["model_state_dict"])
        optimizer.load_state_dict(check["optimizer"])
        print("Load trained model from iteration {}".format(iter))
        

    if not os.path.isfile(mem_path):
        replay_memory = []
    else:
        with open(mem_path, "rb") as f:
            replay_memory = pickle.load(f)
        print("Load replay memory")
        
    require = nn.MSELoss()
    env = ChromeDino()
    state, _, _ = env.step(0)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
    while iter < input.num_iters:
        
        prediction = model(state)[0]
        
        # Exploration or exploitation
        epsilon = input.final_epsilon + (
            max(input.num_decay_iters - iter, 0) * (input.initial_epsilon - input.final_epsilon) / input.num_decay_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            action = randint(0, 2)
        else:
            action = torch.argmax(prediction).item()

        next_state, reward, done = env.step(action)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        replay_memory.append([state, action, reward, next_state, done])
        if len(replay_memory) > input.replay_memory_size:
            del replay_memory[0]
        b = sample(replay_memory, min(
            len(replay_memory), input.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *b)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0, 0] if action == 0 else [0, 1, 0] if action == 1 else [0, 0, 1] for action in
                      action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(
            np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(
            tuple(state for state in next_state_batch))

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if done else reward + input.gamma * torch.max(prediction) for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))
        
        optimizer.zero_grad()
        loss = require(torch.sum(current_prediction_batch * action_batch, dim=1), y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        print("Iteration: {}/{}, Loss: {:.5f}, Epsilon {:.5f}, Reward: {}".format(iter + 1, input.num_iters, loss, epsilon, reward))
        if (iter + 1) % 1000 == 0:
            check = {"iter": iter,"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(check, check_path)
            with open(mem_path, "wb") as f:
                pickle.dump(replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)

        


if __name__ == "__main__":
    input = helper_arg()
    train_func(input)
