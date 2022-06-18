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
            {"params": body.parameters(), "lr": .0001},
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .001}
        ])
        return model, optimizer

    def two_headed_deep_large(self, model_name: str) -> Tuple[acn.ACNet, torch.optim.Optimizer]:
        body = torch.nn.Sequential(
            torch.nn.LazyConv2d(20, 3),
            torch.nn.LeakyReLU(),
            torch.nn.LazyConv2d(30, 5),
            torch.nn.LeakyReLU(),
            torch.nn.LazyConv2d(40, 4),
            torch.nn.LeakyReLU(),
            torch.nn.LazyConv2d(50, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
        )
        actor = torch.nn.Sequential(
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(120),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(150),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(110),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(70),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(self.action_space),
            torch.nn.Softmax(dim=-1)
        )
        critic = torch.nn.Sequential(
            torch.nn.LazyLinear(100),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(50),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(35),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(30),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(30),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(25),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(20),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(15),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(10),
            torch.nn.LeakyReLU(),
            torch.nn.LazyLinear(1)
        )
        model = acn.TwoHeaded(body, actor, critic, model_name)
        model.initialize(self.observation_space)
        optimizer = torch.optim.Adam([
            {"params": body.parameters(), "lr": .0001},
            {"params": actor.parameters(), "lr": .0005},
            {"params": critic.parameters(), 'lr': .001}
        ])
        return model, optimizer


