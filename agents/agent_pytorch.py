import re
from typing import List

import numpy
import torch
from torchinfo import summary

from agents.snake_agent import SnakeAgent
from games.snake import Snake
from . import utils

SAVE_MODEL_LABEL = "model"
SAVE_OPTIMIZER_LABEL = "actor_optimizer"
SAVE_FILE_EXTENSION = ".pt"
SAVED_MODELS_FOLDER = "saved_models/"
EPISODES_BETWEEN_SAVES = 1000


def device_setup():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        cuda_id = torch.cuda.current_device()
        print(f"CUDA current device id: {cuda_id}")
        print(f"CUDA device name: {torch.cuda.get_device_name(cuda_id)}")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compress_model_str(model: torch.nn.Sequential):
    compressed_str = ""
    for layer in model.children():
        layer_str = str(layer)
        # [:-1] Leaves the end parenthesis out
        separated = layer_str[:-1].split("(", 1)
        layer_type, layer_info = separated
        # First letter of the layer type, and open parenthesis
        layer_compressed_str = layer_type[0] + "("
        # Separate the hyper parameters
        # Only split on commas that are not in parentheses
        layer_hyper_params = re.split(r",\s*(?![^()]*\))", layer_info)
        hyper_params_iter = iter(layer_hyper_params)
        hyper_param = next(hyper_params_iter).split("=")
        # Get the second entry if there is an equals, otherwise the value is the string
        if len(hyper_param) > 1:
            hyper_param = hyper_param[1]
        layer_compressed_str += hyper_param[0]
        for hyper_param in hyper_params_iter:
            hyper_param_split = hyper_param.split("=")
            if len(hyper_param_split) > 1:
                hyper_param = hyper_param_split[1]
            layer_compressed_str += "," + hyper_param
        # Tailing parenthesis to match the first
        layer_compressed_str += ")"
        # Add to the string
        compressed_str += layer_compressed_str
    return compressed_str


def compress_optimizer_str(optimizer: torch.optim.Optimizer):
    split_info = str(optimizer).split("\n")
    info_iter = iter(split_info)
    optimizer_name = next(info_iter).split(" ")[0]  # <OptimizerName> (
    next(info_iter)  # Parameter Group 0
    compressed = optimizer_name + "("
    # Do first iteration manually, to prevent leading comma
    key, val = next(info_iter).split(": ")
    compressed += val
    for param in info_iter:
        if param == ")":
            break
        key, val = param.split(": ")
        compressed += "," + val
    compressed += ")"
    return compressed


def calculate_discounted_rewards(rewards: List[float], discount_factor: float) -> List[float]:
    discounted_reward = 0.
    discounted_rewards = [0.] * len(rewards)
    for time_step, reward in zip(reversed(range(len(rewards))), reversed(rewards)):
        discounted_reward = discount_factor * discounted_reward + reward
        discounted_rewards[time_step] = discounted_reward
    return discounted_rewards


class AgentPyTorch(SnakeAgent):
    def __init__(self, snake_game: Snake, agent_name="", actor=None):
        super().__init__(snake_game)
        self.device = device_setup()
        self.agent_name = agent_name
        observation_space = 1, 1, self.snake_game.get_grid_height(
        ), self.snake_game.get_grid_width()
        self.actor = actor
        if self.actor is None:
            self.actor = self.construct_actor(observation_space, 4)
        summary(self.actor, observation_space)
        self.rewards_sum = 0
        self.best_reward = 0
        self.average_rewards = 0

    def construct_actor(self, obs_space, action_space) -> torch.nn.Sequential:
        raise NotImplementedError

    def collect_experience(self, state: torch.Tensor, distribution: torch.distributions.Categorical, action: int):
        raise NotImplementedError

    def get_action(self, observation: numpy.ndarray, training=True) -> int:
        state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
        # Sample action
        probabilities = self.actor(state)
        dist = torch.distributions.Categorical(probs=probabilities)
        action = dist.sample().item()
        if training:
            self.collect_experience(state, dist, action)
        return action

    def train_agent(self):
        pass

    def reset_agent(self):
        pass

    def give_reward(self, reward: float):
        pass

    def reset(self):
        super().reset()
        # Pass an initial state to the agent, and get an initial action
        observation = self.get_observation()
        self.take_action(self.get_action(observation, self.training))

    def step(self):
        super().step()
        # Get the reward for taking the previous action and transition into this new state,
        # pass the new state, get a new action
        reward = self.get_reward()
        self.give_reward(reward)
        self.rewards_sum += reward
        observation = self.get_observation()
        self.take_action(self.get_action(observation, self.training))

    def end_of_episode(self):
        super().end_of_episode()
        # Obtain the reward for the last action made
        reward = self.get_reward()
        self.give_reward(reward)
        self.rewards_sum += reward
        if self.training:
            self.train_agent()
        self.reset_agent()

        # Statistics
        if self.plotting:
            self.rewards_plot.plot_data(self.rewards_sum)
        if self.rewards_sum > self.best_reward:
            self.best_reward = self.rewards_sum
            print(
                f"New highest rewards collected in one game: {self.best_reward}")
        games_played = self.snake_game.get_games_played()
        self.average_rewards = (
            self.average_rewards * games_played + self.rewards_sum) / (games_played + 1)
        self.rewards_sum = 0

        # Periodic Saving
        if (self.snake_game.get_games_played() + 1) % EPISODES_BETWEEN_SAVES == 0:
            self.save()
            print("Progress Saved")
            print(f"Average reward so far: {self.average_rewards}")

    def save(self):
        """
        Save this agent
        :return: None
        """
        raise NotImplementedError

    def load(self):
        """
        Load this agent from a previous save
        :return: None
        """
        raise NotImplementedError

    def generate_filename(self, model: torch.nn.Sequential, optimizer: torch.optim.Optimizer, filename: str = None):
        if filename is None:
            filename = compress_model_str(
                model) + compress_optimizer_str(optimizer)
        filename = f"{utils.get_current_directory()}" \
                   f"{SAVED_MODELS_FOLDER}({self.agent_name}){filename}{SAVE_FILE_EXTENSION} "
        return filename

    def save_model(self, model: torch.nn.Sequential, optimizer: torch.optim.Optimizer, filename=None):
        """
        Save a model and its actor_optimizer to a file
        :param model: The model to save
        :param optimizer: The actor_optimizer of the model
        :param filename: The name of the file, set to None to generate a name based on the model parameters
        :return: None
        """
        to_save = {SAVE_MODEL_LABEL: model.state_dict(
        ), SAVE_OPTIMIZER_LABEL: optimizer.state_dict()}
        filename = self.generate_filename(model, optimizer, filename)
        torch.save(to_save, filename)

    def load_model(self, model: torch.nn.Sequential, optimizer: torch.optim.Optimizer, filename=None):
        """
        Load a model and its actor_optimizer from a file generated by the save_model function
        :param model: The model to load into
        :param optimizer: The actor_optimizer of the model to load into
        :param filename: The name of the file to load from
        :return: None
        """
        filename = self.generate_filename(model, optimizer, filename)
        try:
            loaded = torch.load(filename, map_location=self.device)
            model.load_state_dict(loaded[SAVE_MODEL_LABEL])
            optimizer.load_state_dict(loaded[SAVE_OPTIMIZER_LABEL])
        except FileNotFoundError:
            print(f"Could not load model, file not found: \n{filename}")
