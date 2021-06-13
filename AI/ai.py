import Snake.snake as snake


class Ai:
    game = None

    def init_vars(self):
        pass

    def update(self):
        pass

    def __init__(self, game):
        self.game = game
        self.init_vars()
        game.add_routine_update(self.update, [])
