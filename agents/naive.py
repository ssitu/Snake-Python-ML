from agents.snake_agent import SnakeAgent
from games.snake import Snake


class Naive(SnakeAgent):
    input_up = 0
    input_left = 1
    input_down = 2
    input_right = 3
    path = None
    path_alts = None
    path_queue = None
    path_queue_index = 0
    head_location = [-1, -1]

    def path_even_rows(self, grid_rows, grid_cols):
        path = [[0 for _ in range(0, grid_cols)] for _ in range(0, grid_rows)]
        # Straightaway but right at the top
        path[1][1] = self.input_right
        for row in range(2, grid_rows - 1):
            path[row][1] = self.input_up
        # Every other row, right until the edge and then down until the last row
        for row in range(1, grid_rows - 1, 2):
            for col in range(2, grid_cols - 2):
                path[row][col] = self.input_right
            path[row][grid_cols - 2] = self.input_down
        # Every other row after the first, down next to the Straightaway, and left until the last column
        # and then down until the last row
        for row in range(2, grid_rows - 1, 2):
            path[row][2] = self.input_down
            for col in range(3, grid_cols - 1):
                path[row][col] = self.input_left
        # Last row is all lefts to make it back to the Straightaway
        for col in range(2, grid_cols - 1):
            path[grid_rows - 2][col] = self.input_left
        return path

    def path_even_cols(self, grid_rows, grid_cols):
        path = [[0 for _ in range(0, grid_cols)] for _ in range(0, grid_rows)]
        # Straightaway but right at the top
        path[1][1] = self.input_down
        for row in range(2, grid_cols - 1):
            path[1][row] = self.input_left
        # Every other row, right until the edge and then down until the last row
        for row in range(1, grid_cols - 1, 2):
            for col in range(2, grid_rows - 2):
                path[col][row] = self.input_down
            path[grid_rows - 2][row] = self.input_right
        # Every other row after the first, down next to the Straightaway, and left until the last column
        # and then down until the last row
        for row in range(2, grid_cols - 1, 2):
            path[2][row] = self.input_right
            for col in range(3, grid_rows - 1):
                path[col][row] = self.input_up
        # Last row is all lefts to make it back to the Straightaway
        for col in range(2, grid_rows - 1):
            path[col][grid_cols - 2] = self.input_up
        return path

    def path_odd_1(self, grid_rows, grid_cols):
        path = [[0 for _ in range(0, grid_cols)] for _ in range(0, grid_rows)]
        # Left Straightaway up to the first row
        for row in range(2, grid_rows - 1):
            path[row][1] = self.input_up
        # Top Straightaway right and down at the end
        for col in range(1, grid_cols - 2):
            path[1][col] = self.input_right
        path[1][grid_cols - 2] = self.input_down
        # Down again and loop around upwards to the 2nd row, leaving a 1x1 hole
        path[2][grid_cols - 2] = self.input_down
        path[3][grid_cols - 2] = self.input_left
        path[3][grid_cols - 3] = self.input_left
        path[3][grid_cols - 4] = self.input_up
        path[2][grid_cols - 4] = self.input_left
        # Fill rows 2 and 3 by zigzaging
        # Alternating lefts in row 2
        for col in range(3, grid_cols - 5, 2):
            path[2][col] = self.input_left
        # Alternating downs in row 2
        for col in range(2, grid_cols - 4, 2):
            path[2][col] = self.input_down
        # Alternating lefts in row 3
        for col in range(4, grid_cols - 4, 2):
            path[3][col] = self.input_left
        # Alternating ups in row 3
        for col in range(3, grid_cols - 5, 2):
            path[3][col] = self.input_up
        # One last down next to the upward Straightaway on row 3
        path[3][2] = self.input_down
        # Right rows every other
        for row in range(4, grid_rows - 2, 2):
            for col in range(2, grid_cols - 2):
                path[row][col] = self.input_right
            # Down at the end of the row
            path[row][grid_cols - 2] = self.input_down
        # Left rows every other except the end row
        for row in range(5, grid_rows - 2, 2):
            for col in range(3, grid_cols - 1):
                path[row][col] = self.input_left
            # Down at the front of the row
            path[row][2] = self.input_down
        # Left Straightaway towards the up Straightaway
        for col in range(2, grid_cols - 1):
            path[grid_rows - 2][col] = self.input_left
        return path

    def path_odd_2(self, grid_rows, grid_cols):
        path = [[0 for _ in range(0, grid_cols)] for _ in range(0, grid_rows)]
        # Left Straightaway up to the first row
        for row in range(2, grid_rows - 1):
            path[row][1] = self.input_up
        # Top Straightaway right and down at the end
        for col in range(1, grid_cols - 2):
            path[1][col] = self.input_right
        path[1][grid_cols - 2] = self.input_down
        # Alternating lefts on row 2
        for col in range(3, grid_cols - 1, 2):
            path[2][col] = self.input_left
        # Alternating downs on row 2
        for col in range(2, grid_cols - 2, 2):
            path[2][col] = self.input_down
        # Alternating ups on row 3
        for col in range(3, grid_cols - 2, 2):
            path[3][col] = self.input_up
        # Alternating lefts on row 3
        for col in range(4, grid_cols - 2, 2):
            path[3][col] = self.input_left
        # Down after the rightmost down in row 2
        path[3][2] = self.input_down
        # Right rows every other
        for row in range(4, grid_rows - 2, 2):
            for col in range(2, grid_cols - 2):
                path[row][col] = self.input_right
            # Down at the end of the row
            path[row][grid_cols - 2] = self.input_down
        # Left rows every other except the end row
        for row in range(5, grid_rows - 2, 2):
            for col in range(3, grid_cols - 1):
                path[row][col] = self.input_left
            # Down at the front of the row
            path[row][2] = self.input_down
        # Left Straightaway towards the up Straightaway
        for col in range(2, grid_cols - 1):
            path[grid_rows - 2][col] = self.input_left

        return path

    @staticmethod
    def print_matrix(matrix):
        for row in matrix:
            for col in row:
                print(col, end=" ")
            print()

    def __init__(self, env: Snake):
        super().__init__(env)
        grid_rows = self.snake_game.get_grid_height()
        grid_cols = self.snake_game.get_grid_width()
        # Layout the Hamiltonian path
        # Even rows
        if grid_rows % 2 == 0:
            self.path = self.path_even_rows(grid_rows, grid_cols)
            print("Path:")
            self.print_matrix(self.path)
        # Even columns
        elif grid_cols % 2 == 0:
            self.path = self.path_even_cols(grid_rows, grid_cols)
            print("Path:")
            self.print_matrix(self.path)
        # Both rows and columns are odd
        else:
            self.path_queue = [self.path_odd_1(grid_rows, grid_cols), self.path_odd_2(grid_rows, grid_cols)]
            self.path = self.path_queue[0]
            print("Odd rows and columns, must use two paths")
            print("Path 1:")
            self.print_matrix(self.path_queue[0])
            print("Path 2:")
            self.print_matrix(self.path_queue[1])

    def step(self):
        super().step()
        self.head_location = self.snake_game.get_head_location()
        # Start of algorithm
        # Switch path when at grid location (1, 1)
        if self.path_queue is not None and self.head_location[0] == 1 and self.head_location[1] == 1:
            self.path_queue_index = (self.path_queue_index + 1) % len(self.path_queue)
            self.path = self.path_queue[self.path_queue_index]
        instruction = self.path[self.head_location[0]][self.head_location[1]]
        if instruction == self.input_up:
            self.snake_game.set_input_up()
        elif instruction == self.input_left:
            self.snake_game.set_input_left()
        elif instruction == self.input_down:
            self.snake_game.set_input_down()
        else:
            self.snake_game.set_input_right()

    def end_of_episode(self):
        super().end_of_episode()
        # Nothing to do

    def reset(self):
        self.snake_game.set_input_down()  # Start off moving vertically,
        # the snake can get stuck moving right until hitting the edge on the path where it should be moving left
