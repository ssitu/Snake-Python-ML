from games.game import Game


class Agent:

    def __init__(self, env: Game):
        self.env = env
        self.env.add_routine_update(self.update, [])

    def update(self):
        pass

