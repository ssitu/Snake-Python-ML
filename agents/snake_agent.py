import numpy

from agents.agent import Agent
from agents.plot import Plot
from games.snake import Snake

INPUT_UP = 0
INPUT_DOWN = 1
INPUT_LEFT = 2
INPUT_RIGHT = 3


class SnakeAgent(Agent):
    """
    Handles Snake specific flags and logic
    """

    def __init__(self, env: Snake):
        super().__init__(env)
        self.snake_game = env
        self.rewards_plot = Plot("Agent Performance", "Games Played", "Rewards", moving_average_length=10000)
        self.plotting = False
        self.training = True
        self.game_area = self.snake_game.get_grid_width() * self.snake_game.get_grid_height()

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

    def get_observation(self):
        return numpy.expand_dims(numpy.array(self.snake_game.get_grid()), axis=0)

    def get_reward(self):
        if self.snake_game.is_state_lose():
            return -1
        if self.snake_game.is_state_eaten_apple():
            return 1
        if self.snake_game.is_state_win():
            return self.game_area
        return - 1 / self.game_area

    def take_action(self, action: int):
        if action == INPUT_UP:
            self.snake_game.set_input_up()
        elif action == INPUT_DOWN:
            self.snake_game.set_input_left()
        elif action == INPUT_LEFT:
            self.snake_game.set_input_down()
        elif action == INPUT_RIGHT:
            self.snake_game.set_input_right()
        else:
            raise RuntimeWarning("Invalid Input")

    def start_plot(self):
        self.rewards_plot.start_updates()
        self.plotting = True

    def stop_plot(self):
        self.rewards_plot.stop_updates()
        self.plotting = False

    def save(self):
        pass

    def load(self):
        pass

    def set_training(self, condition: bool):
        """
        Set the condition to turn on or off training for this agent
        :param condition: The boolean to determine if the agent should train
        :return: None
        """
        self.training = condition
