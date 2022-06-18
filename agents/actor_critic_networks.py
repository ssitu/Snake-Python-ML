from typing import Tuple

import torch.nn
from torchinfo import torchinfo

from agents import utils

utils.setup_torch()


class ACNet(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, obs: torch.tensor) -> Tuple[int, float]:
        pass

    def initialize(self, input_shape):
        torchinfo.summary(self, input_shape)


class Separate(ACNet):
    def __init__(self, actor: torch.nn.Sequential, critic: torch.nn.Sequential, name: str):
        super().__init__(name)
        self.actor = actor
        self.critic = critic

    def forward(self, obs: torch.tensor) -> Tuple[int, float]:
        return self.actor(obs), self.critic(obs)


class TwoHeaded(ACNet):
    def __init__(self, body: torch.nn.Sequential, actor: torch.nn.Sequential, critic: torch.nn.Sequential, name: str):
        super().__init__(name)
        self.body = body
        self.actor = actor
        self.critic = critic

    def forward(self, obs: torch.tensor) -> Tuple[int, float]:
        x = self.body(obs)
        state_value = self.critic(x)
        prob_dist = self.actor(x)
        return prob_dist, state_value
