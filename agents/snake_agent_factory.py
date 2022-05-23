from agents.naive import Naive
from agents.snake_agent import SnakeAgent
from games.snake import Snake


class SnakeAgentFactory:

    def __init__(self, snake_game: Snake):
        self.snake_game = snake_game

    def create_naive(self) -> SnakeAgent:
        return Naive(self.snake_game)
