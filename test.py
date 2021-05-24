"""
AI Pract Final Project
"""
import argparse
import torch

from src.Agent import Agent
from src.env import ChromeDino
import cv2


def helper_arg():
    parameters = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Chrome Dino""")
    parameters.add_argument("--saved_path", type=str, default="trained_models")
    parameters.add_argument("--fps", type=int, default=60, help="frames per second")
    parameters.add_argument("--output", type=str,
                     default="output/chrome_dino.mp4", help="the path to output video")

    resulting_arg = parameters.parse_args()
    return resulting_arg


def test_func(input):
    torch.manual_seed(123)
    model = Agent()
    check = torch.load("{}/chrome_dino.pth".format(input.saved_path), map_location='cpu')
    model.load_state_dict(check["model_state_dict"])
    model.eval()
    environment = ChromeDino()
    state, raw_state, _, _ = environment.step(0, True)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
    output = cv2.VideoWriter(input.output, cv2.VideoWriter_fourcc(
        *"MJPG"), input.fps, (600, 150))
    exit = False
    while not exit:
        action = torch.argmax(model(state)[0]).item()
        next_state, raw_next_state, reward, exit = environment.step(
            action, True)
        output.write(raw_next_state)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        state = next_state


if __name__ == "__main__":
    input = helper_arg()
    test_func(input)
