import numpy as np
from tensorflow import keras
import AI.tensorflow_ai as tfai


class Averaging(tfai.Ai_tf):

    model_name = "model_averaging"

    def create_model(self):
        # grid = self.game.get_grid()
        # grid_rows = len(grid)
        # grid_cols = len(grid[0])
        # input_shape = (1, grid_rows, grid_cols, 1)  # Batch size, rows, columns, depth
        model = keras.models.Sequential()
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(50))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.Dense(50))
        model.add(keras.layers.LeakyReLU(alpha=0.1))
        model.add(keras.layers.Dense(4, activation="sigmoid"))
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics='accuracy')
        return model

    def init_vars(self):
        self.model = self.create_model()
        self.state_history = []
        self.action_history = []
        self.head_location = None

    def update(self):
        current_head_location = self.game.get_head_location()
        # Check if the snake has moved to avoid making repetitive decisions
        if self.head_location != current_head_location:
            self.head_location = current_head_location
            # Start of algorithm
            grid = np.array(self.game.get_grid())
            input = np.expand_dims(np.expand_dims(grid, axis=0), axis=3)
            self.state_history.append(input)
            prediction = np.argmax(self.model.predict(input))
            self.action_history.append(prediction)
            if prediction == 0:
                self.game.set_input_up()
            elif prediction == 1:
                self.game.set_input_down()
            elif prediction == 2:
                self.game.set_input_left()
            else:
                self.game.set_input_right()
