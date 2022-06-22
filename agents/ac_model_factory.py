from typing import Tuple

import torch.nn

import agents.actor_critic_networks as acn
from games.snake import Snake


class ModelFactory:

    def __init__(self, env: Snake):
        self.env = env
        self.action_space = 4
        self.observation_space = 1, 1, self.env.get_grid_width(), self.env.get_grid_height()

    def two_headed_deep_small(self, model_name: str) -> Tuple[acn.ACNet, torch.optim.Optimizer]:
        body = torch.nn.Sequential(
            torch.nn.LazyConv2d(30, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
        )
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.action_space),
            torch.nn.Softmax(dim=-1)
        )
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1)
        )
        model = acn.TwoHeaded(body, actor, critic, model_name)
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .00005},
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return model, optimizer

    def two_headed_deep_large(self, model_name: str) -> Tuple[acn.ACNet, torch.optim.Optimizer]:
        body = torch.nn.Sequential(
            torch.nn.LazyConv2d(10, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.LazyConv2d(30, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.LazyConv2d(40, 3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.LazyConv2d(100, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
        )
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.action_space),
            torch.nn.Softmax(dim=-1)
        )
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1)
        )
        model = acn.TwoHeaded(body, actor, critic, model_name)
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .00005},
            {"params": actor.parameters(), "lr": .0001},
            {"params": critic.parameters(), 'lr': .0005}
        ])
        return model, optimizer
