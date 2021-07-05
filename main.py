import Snake.snake as snake
from AI.naive import Naive
from AI.averaging import Averaging

snake = snake.Snake(3, 3)
snake.set_snake_movement_mode_by_input()
# snake.set_max_speed()
snake.set_snake_move_limit_per_apple(True)
agents = {
    0: Naive(),
    1: Averaging(),
 }
active_agent = agents[0]
active_agent.activate(snake)
snake.start()
