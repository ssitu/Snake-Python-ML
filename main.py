import Snake.snake as snake
from AI.naive import Naive
from AI.averaging import Averaging

snake = snake.Snake()
# snake.set_max_speed()
# naive = Naive(snake)
averaging = Averaging(snake)
snake.start()
