from agents.snake_agent_factory import SnakeAgentFactory
from games import snake

snake = snake.Snake(4, 4)
snake.set_max_speed()
snake.set_snake_move_limit_per_apple(True)
agent_factory = SnakeAgentFactory(snake)
agents = {
    0: agent_factory.create_naive
 }
active_agent = agents[0]()
snake.start(games_to_play=1)
