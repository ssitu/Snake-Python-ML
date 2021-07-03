class Ai:
    game = None

    # To be implemented in inheriting classes
    def init_vars(self):
        pass

    # To be implemented in inheriting classes
    def update(self):
        pass

    def __init__(self, game):
        self.game = game
        self.init_vars()
        game.add_routine_update(self.update, [])
