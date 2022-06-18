def main():
    from agents.snake_agent_factory import SnakeAgentFactory
    from games.snake import Snake

    training = True

    snake = Snake(4, 4, render=not training)
    snake.set_print_win(False)
    if training:
        snake.set_max_speed()
    snake.set_snake_move_limit_per_apple(True)
    agent_factory = SnakeAgentFactory(snake)
    agents = {
        0: agent_factory.naive,
        1: agent_factory.ppo_two_head_small,
        2: agent_factory.ppo_two_head_large,
    }
    active_agent = agents[1]()  # Pick and construct the agent
    active_agent.set_training(training)
    active_agent.load()
    active_agent.start_plot()
    snake.start()
    active_agent.stop_plot()
    if training:
        active_agent.save()


if __name__ == '__main__':
    main()
