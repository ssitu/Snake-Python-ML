from agents.agent import Agent
from games.snake import Snake


class SnakeAgent(Agent):
    """
    Handles Snake specific flags and logic
    """

    def __init__(self, env: Snake):
        super().__init__(env)
        self.snake_game = env

    def update(self):
        """
        Handles the snake specific flags
        :return:
        """
        super().update()
        if self.snake_game.is_state_moved_previous_frame():
            if not self.snake_game.is_state_win() and not self.snake_game.is_state_lose():
                self.step()
            else:
                self.end_of_episode()

    def step(self):
        """
        Is called when a new frame has occurred
        :return: None
        """
        pass

    def end_of_episode(self):
        """
        Is called at the end of the game
        :return: None
        """
        pass


