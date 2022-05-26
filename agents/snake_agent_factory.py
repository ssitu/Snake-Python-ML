from agents.naive import Naive
from agents.ppo import AgentPPO
from agents.snake_agent import SnakeAgent
from games.snake import Snake


class SnakeAgentFactory:

    def __init__(self, snake_game: Snake):
        self.snake_game = snake_game

    def create_naive(self) -> SnakeAgent:
        return Naive(self.snake_game)

    def create_ppo(self) -> SnakeAgent:
        name = f"AgentPPO{self.snake_game.get_grid_width()-2}x{self.snake_game.get_grid_height()-2}"
        return AgentPPO(self.snake_game, agent_name=name)
