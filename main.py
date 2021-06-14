import Snake.snake as snake
import AI.naive as naive

snake = snake.Snake()
snake.set_max_speed()
naive = naive.Naive(snake)
snake.start()
