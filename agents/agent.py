class Agent:
    game = None

    # To be implemented in inheriting classes
    def init_vars(self):
        pass

    # To be implemented in inheriting classes
    def update(self):
        pass

    def activate(self, game):
        self.game = game
        self.init_vars()
        self.game.add_routine_update(self.update, [])
