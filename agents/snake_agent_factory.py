import torch

from agents.ac_model_factory import ModelFactory
from agents.actor_critic_networks import ACNet
from agents.naive import Naive
from agents.ppo import AgentPPO
from agents.snake_agent import SnakeAgent
from games.snake import Snake


class SnakeAgentFactory:

    def __init__(self, snake_game: Snake):
        self.snake_game = snake_game
        self.model_factory = ModelFactory(snake_game)

    def naive(self) -> SnakeAgent:
        return Naive(self.snake_game)

    def ppo_two_head_small(self) -> SnakeAgent:
        name = f"PPO{self.snake_game.get_grid_width()-2}x{self.snake_game.get_grid_height()-2}"
        model, optimizer = self.model_factory.two_headed_deep_small(name)
        return AgentPPO(self.snake_game, model=model, optimizer=optimizer)

    def ppo_two_head_large(self) -> SnakeAgent:
        name = f"PPO{self.snake_game.get_grid_width()-2}x{self.snake_game.get_grid_height()-2}"
        model, optimizer = self.model_factory.two_headed_deep_large(name)
        return AgentPPO(self.snake_game, model=model, optimizer=optimizer)
