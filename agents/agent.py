from games.game import Game


class Agent:

    def __init__(self, env: Game):
        self.env = env
        self.env.add_routine_update(self.update, [])
        self.env.add_routine_reset(self.reset, [])

    def update(self):
        pass

    def reset(self):
        pass

