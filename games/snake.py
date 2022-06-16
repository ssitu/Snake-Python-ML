import math
import random

import pygame

import games.game as game


class Snake(game.Game):
    # Colors
    _color_empty_rgb = (21, 21, 21)
    _color_snake_head_rgb = (154, 205, 50)
    _color_snake_body_rgb = (46, 139, 87)
    _color_apple_rgb = (220, 20, 60)
    _color_empty = pygame.Color(_color_empty_rgb)
    _color_snake_head = pygame.Color(_color_snake_head_rgb)
    _color_snake_body = pygame.Color(_color_snake_body_rgb)
    _color_apple = pygame.Color(_color_apple_rgb)
    # Graphics related
    _pixels_screen_size = 768
    # Time related
    _time_movement_delay_secs_default = .13
    _time_movement_delay_secs = _time_movement_delay_secs_default
    _time_since_last_movement_secs = 0
    # Grid related
    _grid_cell_empty = 0
    _grid_cell_snake_body_range = (1, 3)
    _grid_cell_snake_body_range_size = _grid_cell_snake_body_range[1] - _grid_cell_snake_body_range[0]
    _grid_cell_snake_head = 5
    _grid_cell_apple = 10
    # Directions for the games to move in
    _direction_up = (-1, 0)
    _direction_down = (1, 0)
    _direction_left = (0, -1)
    _direction_right = (0, 1)
    # games related
    _snake_list = []
    _snake_direction = _direction_right
    _snake_direction_next = _snake_direction
    _snake_movement_mode_by_delay = 0
    _snake_movement_mode_by_input = 1
    _snake_movement_mode = _snake_movement_mode_by_delay
    # Key inputs
    _input_nothing = 0
    _input_up = 1
    _input_down = 2
    _input_left = 3
    _input_right = 4
    _input = _input_nothing
    # States
    _states = 0
    _state_eaten_apple = 1
    _state_hit_body = 1 << 1
    _state_hit_edge = 1 << 2
    _state_win = 1 << 3
    _state_move_limit_per_apple_reached = 1 << 4
    _state_moved_previous_frame = 1 << 5

    def __init__(self, grid_width=17, grid_height=17, snake_pixels_border_percentage=1 / 6, render=True):
        print("Snake initialization started.")
        game.Game.__init__(self, self._pixels_screen_size, self._pixels_screen_size, render)
        self.game_screen.fill(self._color_empty)
        # Add two to the grid width and height to add a buffer around the grid to show end states
        # when the head moves out of bounds
        self._grid_width = grid_width + 2
        self._grid_visible_width = grid_width
        self._grid_height = grid_height + 2
        self._grid_visible_height = grid_height
        print(f"Grid Width: {grid_width}, Grid Height: {grid_height}")
        self._snake_move_limit_per_apple = -1
        self._grid = [[0 for col in range(self._grid_width)] for row in range(self._grid_height)]
        self._pixels_unit_width = self._pixels_screen_size / grid_width
        self._pixels_unit_height = self._pixels_screen_size / grid_height
        self._pixels_border_percentage = snake_pixels_border_percentage
        self._pixels_border_offset_x = self._pixels_unit_width * self._pixels_border_percentage / 2
        self._pixels_border_offset_y = self._pixels_unit_height * self._pixels_border_percentage / 2
        self._pixels_unit_center_width = self._pixels_unit_width * (1 - self._pixels_border_percentage)
        self._pixels_unit_center_height = self._pixels_unit_height * (1 - self._pixels_border_percentage)
        if self.render:
            pygame.display.flip()
        self.set_speed(- 6 / (grid_width * grid_height) + 1)
        self.games_played = -1  # Is added to after a reset, -1 to leave out the first reset
        print("Snake initialization finished.")

    def _print_grid(self):
        print()
        for row in self._grid:
            for col in row:
                print(col, end=" ")
            print()

    def _set_grid_fill(self, row, col, pygame_color):
        # Must subtract one to the coordinates due to the hidden border around the outside of the game window
        start_x = (col - 1) * self._pixels_unit_width
        start_y = (row - 1) * self._pixels_unit_height
        center_x = start_x + self._pixels_border_offset_x
        center_y = start_y + self._pixels_border_offset_y
        rectangle = pygame.Rect(center_x, center_y, self._pixels_unit_center_width, self._pixels_unit_center_height)
        if self.render:
            pygame.draw.rect(self.game_screen, pygame_color, rectangle)

    def _set_cell_empty(self, row, col):
        self._grid[row][col] = self._grid_cell_empty
        self._set_grid_fill(row, col, self._color_empty)

    def _set_cell_snake_head(self, row, col):
        self._grid[row][col] = self._grid_cell_snake_head
        self._set_grid_fill(row, col, self._color_snake_head)

    # rgb_base and rgb_to_blend should be an indexable structure of 3 numbers corresponding to rgb values
    # A blend weight of 1 will return rgb_base, A blend weight of 0 will return rgb_to_blend
    # A blend weight of 0.5 will return the average between rgb_base and rgb_to_blend
    @staticmethod
    def _blend_colors(blend_weight, rgb_base, rgb_to_blend):
        blended = []
        for i in range(3):
            blended.append(math.ceil((blend_weight * rgb_base[i] + (1 - blend_weight) * rgb_to_blend[i])))
        return blended

    def _is_cell_snake_body(self, grid_cell_value):
        return self._grid_cell_snake_body_range[0] < grid_cell_value < self._grid_cell_snake_body_range[1]

    def _set_cell_snake_body(self, row, col, body_index):
        weight = 1 - body_index / len(self._snake_list) * (2 / 3)
        # The last fraction is the portion of the upper side of the interval [0, 1] to use
        self._grid[row][col] = self._grid_cell_snake_body_range_size * weight + self._grid_cell_snake_body_range[0]
        blended_body_color = pygame.Color(self._blend_colors(weight, self._color_snake_body_rgb, self._color_empty_rgb))
        self._set_grid_fill(row, col, blended_body_color)

    def _set_cell_apple(self, row, col):
        self._grid[row][col] = self._grid_cell_apple
        self._set_grid_fill(row, col, self._color_apple)

    def _place_starting_snake(self):
        start_row = int(self._grid_height / 2)
        start_col = int(self._grid_width / 2)
        self._set_cell_snake_head(start_row, start_col)
        self._snake_list.append([start_row, start_col])
        self._set_cell_snake_body(start_row, start_col - 1, 0)
        self._snake_list.append([start_row, start_col - 1])
        self._set_cell_snake_body(start_row, start_col - 2, 1)
        self._snake_list.append([start_row, start_col - 2])
        self._set_cell_snake_body(start_row, start_col - 3, 2)
        self._snake_list.append([start_row, start_col - 3])

    def _routine_reset_game(self):
        self._snake_list = []
        self._snake_direction = self._direction_right
        self._snake_direction_next = self._snake_direction
        self._grid = [[0 for _ in range(self._grid_width)] for _ in range(self._grid_height)]
        background_rectangle = pygame.Rect(0, 0, self._pixels_screen_size, self._pixels_screen_size)
        pygame.draw.rect(self.game_screen, self._color_empty, background_rectangle)
        self._place_starting_snake()
        self._routine_apple_spawn()
        self._routine_reset()  # Call external reset functions
        self.games_played += 1

    def _is_grid_full(self):
        for row in range(1, self._grid_visible_height + 1):
            for col in range(1, self._grid_visible_width + 1):
                if self._grid[row][col] == self._grid_cell_empty:
                    return False
        return True

    def _is_grid_full_by_snake(self):
        for row in range(1, self._grid_visible_height + 1):
            for col in range(1, self._grid_visible_width + 1):
                cell = self._grid[row][col]
                # Assuming that empty and apple cells are the only cells other than the games body and head cells
                if cell == self._grid_cell_empty or cell == self._grid_cell_apple:
                    return False
        return True

    def _routine_apple_spawn(self, next_head_location_row=None, next_head_location_col=None):
        if not self._is_grid_full():
            # Pick a random location on the visible part of the grid
            random_row = random.randint(1, self._grid_visible_height)
            random_col = random.randint(1, self._grid_visible_width)
            # Keep getting a new location until the random location is empty
            while self._grid[random_row][random_col] != self._grid_cell_empty:
                random_row = random.randint(1, self._grid_visible_height)
                random_col = random.randint(1, self._grid_visible_width)
            self._set_cell_apple(random_row, random_col)
            self._snake_move_count_per_apple = 0

    # Call after the new direction has been determined but before games moves in the next determined location
    def _routine_apple_eaten(self, next_head_location_row, next_head_location_col):
        will_eat_apple = self._grid[next_head_location_row][next_head_location_col] == self._grid_cell_apple
        if will_eat_apple:
            self._states = self._states | self._state_eaten_apple
            # Add a body cell to the games at the tail location
            tail_location = self._snake_list[-1]
            self._snake_list.append([tail_location[0], tail_location[1]])
            # No other logic is needed:
            # the new head pixels replace the apple pixels
            # the new head cell value replaces the grid cell value for the apple
            self._routine_apple_spawn(next_head_location_row, next_head_location_col)

    # Call after the new direction has been determined but before games moves in the next determined location
    def _routine_collision_edges(self, next_head_location_row, next_head_location_col):
        # Must compensate for the grid's extra row and column
        will_collide_horizontal_edges = next_head_location_row > self._grid_visible_height or next_head_location_row < 1
        will_collide_vertical_edges = next_head_location_col > self._grid_visible_width or next_head_location_col < 1
        will_collide_edge = will_collide_horizontal_edges or will_collide_vertical_edges
        if will_collide_edge:
            self._states = self._states | self._state_hit_edge

    # Call after the new direction has been determined but before games moves in the next determined location
    def _routine_collision_body(self, next_head_location_row, next_head_location_col):
        will_collide_body = False
        try:
            will_collide_body = self._is_cell_snake_body(self._grid[next_head_location_row][next_head_location_col])
        except IndexError:
            print(next_head_location_row, next_head_location_col)
            print(self._states)
            self._print_grid()
        if will_collide_body:
            tail = self._snake_list[-1]
            body_is_tail = next_head_location_row == tail[0] and next_head_location_col == tail[1]
            if not body_is_tail:
                self._states = self._states | self._state_hit_body

    def _routine_snake_move(self, next_head_location_row, next_head_location_col):
        cell_next = self._snake_list[0].copy()
        # Move head
        self._snake_list[0][0] = next_head_location_row
        self._snake_list[0][1] = next_head_location_col
        self._set_cell_snake_head(self._snake_list[0][0], self._snake_list[0][1])
        # Move each body part
        body_length = len(self._snake_list)
        for i in range(1, body_length):
            # Save the current location for the next body part
            cell_current = self._snake_list[i].copy()
            # Save the new location for the current body part
            self._snake_list[i][0] = cell_next[0]
            self._snake_list[i][1] = cell_next[1]
            # Set the new location as a body part
            self._set_cell_snake_body(self._snake_list[i][0], self._snake_list[i][1], i - 1)
            # Save the old location for the next body part
            cell_next = cell_current
        # Clear the tail, but only if it is the tail, in the case that the new head is where the tail was
        # Special case when games eats an apple, the new tail would be set as empty
        if self._is_cell_snake_body(self._grid[cell_next[0]][cell_next[1]]) and not self.is_state_eaten_apple():
            self._set_cell_empty(cell_next[0], cell_next[1])
        self._states = self._states | self._state_moved_previous_frame

    def _routine_inputs_keys(self):
        if self._events_key_down:
            if self._events_key_down[-1].key in (pygame.K_w, pygame.K_UP):
                self._input = self._input_up
            elif self._events_key_down[-1].key in (pygame.K_a, pygame.K_LEFT):
                self._input = self._input_left
            elif self._events_key_down[-1].key in (pygame.K_s, pygame.K_DOWN):
                self._input = self._input_down
            elif self._events_key_down[-1].key in (pygame.K_d, pygame.K_RIGHT):
                self._input = self._input_right

    def _routine_inputs(self):
        direction = None
        if self._input == self._input_up:
            direction = self._direction_up
        if self._input == self._input_left:
            direction = self._direction_left
        if self._input == self._input_down:
            direction = self._direction_down
        if self._input == self._input_right:
            direction = self._direction_right
        if direction is not None and self._snake_direction != (direction[0] * -1, direction[1] * -1):
            self._snake_direction_next = direction

    # Call after games has moved
    def _routine_win(self):
        win = self._is_grid_full_by_snake()
        if win:
            self._states = self._states | self._state_win

    def _routine_move_snake_by_delay(self):
        # Movement with set delay
        self._time_since_last_movement_secs += self._game_frame_delay
        if self._time_since_last_movement_secs > self._time_movement_delay_secs or self._time_no_sleep:
            self._routine_next_frame()

    def _routine_move_snake_by_input(self):
        if self._events_key_down:
            self._routine_next_frame()

    def _routine_next_frame(self):
        self._snake_direction = self._snake_direction_next
        next_head_location_row = self._snake_list[0][0] + self._snake_direction[0]
        next_head_location_col = self._snake_list[0][1] + self._snake_direction[1]
        self._routine_collision_edges(next_head_location_row, next_head_location_col)
        self._routine_collision_body(next_head_location_row, next_head_location_col)
        self._routine_apple_eaten(next_head_location_row, next_head_location_col)
        self._routine_snake_move(next_head_location_row, next_head_location_col)
        self._routine_win()
        self._time_since_last_movement_secs = 0
        self._snake_move_count_per_apple += 1
        if self._snake_move_count_per_apple > self._snake_move_limit_per_apple > 0:
            self._states = self._states | self._state_move_limit_per_apple_reached

    def start(self, games_to_play=-1):
        self._routine_reset_game()
        while not self.game_quit and (games_to_play > self.games_played or games_to_play == -1):
            self.update()
            if self.is_state_lose():
                # Nice to see the head move off the screen to indicate collision with the edge
                # Already does it with the extra two rows and columns and compensation in the fill_grid function,
                # but it happens too fast to be able to see so sleep is needed before resetting the game
                self._toggleable_sleep(self._time_movement_delay_secs * 2)
                self._routine_reset_game()
            if self.is_state_win():
                print("Win!")
                # Fill each cell with the color of the snake head to indicate a win
                for row in range(1, self._grid_visible_height + 1):
                    for col in range(1, self._grid_visible_width + 1):
                        self._set_grid_fill(row, col, self._color_snake_head)
                if self.render:
                    pygame.display.flip()
                self._toggleable_sleep(self._time_movement_delay_secs * 2)
                self._routine_reset_game()
            self._states = 0
            self._routine_inputs_keys()
            self._routine_inputs()
            if self._snake_movement_mode == self._snake_movement_mode_by_delay:
                self._routine_move_snake_by_delay()
            else:
                self._routine_move_snake_by_input()

    def set_speed(self, speed_factor):
        self._time_movement_delay_secs /= max(.2, speed_factor)

    def get_length(self):
        return len(self._snake_list)

    def is_state_eaten_apple(self):
        return self._states & self._state_eaten_apple == self._state_eaten_apple

    def is_state_hit_body(self):
        return self._states & self._state_hit_body == self._state_hit_body

    def is_state_hit_edge(self):
        return self._states & self._state_hit_edge == self._state_hit_edge

    def is_state_win(self):
        return self._states & self._state_win == self._state_win

    def is_state_move_limit_reached(self):
        return self._states & self._state_move_limit_per_apple_reached == self._state_move_limit_per_apple_reached

    def is_state_lose(self):
        return self.is_state_hit_body() or self.is_state_hit_edge() or self.is_state_move_limit_reached()

    def is_state_moved_previous_frame(self):
        return self._states & self._state_moved_previous_frame == self._state_moved_previous_frame

    def set_input_up(self):
        self._input = self._input_up

    def set_input_left(self):
        self._input = self._input_left

    def set_input_down(self):
        self._input = self._input_down

    def set_input_right(self):
        self._input = self._input_right

    def get_grid(self):
        return [row[:] for row in self._grid]

    def get_visible_grid(self):
        return [[self._grid[row][col] for col in range(1, self._grid_width)] for row in range(1, self._grid_height)]

    def get_head_location(self):
        return self._snake_list[0].copy()

    def set_snake_move_limit_per_apple(self, on: bool = True):
        """
        If True is given, then the maximum number of moves that
        can be made by the snake will be twice the area of the grid.
        If False is given, then there is no maximum number of moves
        :param on:
        :return:
        """
        if on:
            self._snake_move_limit_per_apple = self._grid_width * self._grid_height * 2
        else:
            self._snake_move_limit_per_apple = -1

    def set_snake_movement_mode_by_delay(self):
        self._snake_movement_mode = self._snake_movement_mode_by_delay

    def set_snake_movement_mode_by_input(self):
        self._snake_movement_mode = self._snake_movement_mode_by_input

    def get_grid_height(self):
        return self._grid_height

    def get_grid_width(self):
        return self._grid_width

    def set_render(self, render: bool):
        self.render = render

    def get_games_played(self):
        return self.games_played
