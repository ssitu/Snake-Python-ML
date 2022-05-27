import torch


def main():
    from agents.snake_agent_factory import SnakeAgentFactory
    from games.snake import Snake

    training = True
    filename = "PPOCNN"
    filename = None

    snake = Snake(4, 4, render=not training)
    snake.set_speed(4)
    if training:
        snake.set_max_speed()
    snake.set_snake_move_limit_per_apple(True)
    agent_factory = SnakeAgentFactory(snake)
    agents = {
        0: (agent_factory.create_naive, [], {}),
        1: (agent_factory.create_ppo, [], {"actor": actor4x4(), "critic": critic4x4()}),
        2: (agent_factory.create_ppo, [], {})
    }
    constructor, args, kwargs = agents[1]
    active_agent = constructor(*args, **kwargs)
    active_agent.set_training(training)
    # active_agent.load(filename=filename)
    # active_agent.start_plot()
    snake.start()
    # active_agent.stop_plot()
    if training:
        active_agent.save(filename=filename)


def actor4x4():
    return torch.nn.Sequential(
        torch.nn.LazyConv2d(10, 2),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(20),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(4),
        torch.nn.Softmax(dim=-1)
    )


def critic4x4():
    return torch.nn.Sequential(
        torch.nn.LazyConv2d(10, 2),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(20),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(10),
        torch.nn.LeakyReLU(),
        torch.nn.LazyLinear(1),
    )


if __name__ == '__main__':
    main()
