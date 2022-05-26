def main():
    from agents.snake_agent_factory import SnakeAgentFactory
    from games.snake import Snake

    snake = Snake(4, 4, render=True)
    # snake.set_max_speed()
    snake.set_snake_move_limit_per_apple(True)
    agent_factory = SnakeAgentFactory(snake)
    agents = {
        0: agent_factory.create_naive,
        1: agent_factory.create_ppo,
    }
    active_agent = agents[1]()
    active_agent.set_training(False)
    active_agent.load()
    # active_agent.start_plot()
    snake.start()
    # active_agent.stop_plot()
    active_agent.save()


if __name__ == '__main__':
    main()
