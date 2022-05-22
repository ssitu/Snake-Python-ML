from snake import snake
from agents.naive import Naive

snake = snake.Snake(10, 10)
snake.set_max_speed()
snake.set_snake_move_limit_per_apple(True)
agents = {
    0: Naive(),
 }
active_agent = agents[0]
active_agent.activate(snake)
snake.start()
