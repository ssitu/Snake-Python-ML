import Snake.game as game
import pygame
import random
import math


class Snake(game.Game):
    # Colors
    _color_empty = pygame.Color("#151515")
    _color_snake_head = pygame.Color("#9ACD32")
    _color_snake_body = pygame.Color("#2E8B57")
    _color_apple = pygame.Color("#DC143C")
    # Graphics related
    _pixels_screen_size = 768
    # Time related
    _time_movement_delay_secs_default = .11
    _time_movement_delay_secs = _time_movement_delay_secs_default
    _time_since_last_movement_secs = 0
    # Grid related
    _grid_cell_empty = 0
    _grid_cell_snake_head = 1
    _grid_cell_snake_body = 2
    _grid_cell_apple = 3
    # Directions for the snake to move in
    _direction_up = (-1, 0)
    _direction_down = (1, 0)
    _direction_left = (0, -1)
    _direction_right = (0, 1)
    # Snake related
    _snake_list = []
    _snake_direction = _direction_right
    _snake_direction_next = _snake_direction
    _snake_status_nothing = 0
    _snake_status_eaten_apple = 1
    _snake_status_hit_body = 2
    _snake_status_hit_edge = 3
    _snake_status = _snake_status_nothing

    def __init__(self, grid_width=17, grid_height=17, snake_pixels_border_percentage=1/6):
        print("Snake initialization started.")
        game.Game.__init__(self, self._pixels_screen_size, self._pixels_screen_size)
        self.game_screen.fill(self._color_empty)
        # Add two to the grid width and height to add a buffer around the grid to show end states
        # when the head moves out of bounds
        self._grid_width = grid_width + 2
        self._grid_visible_width = grid_width
        print("Grid Width:", grid_width, end=" | ")
        self._grid_height = grid_height + 2
        self._grid_visible_height = grid_height
        print("Grid Height:", grid_height, end=" | ")
        self._grid = [[0 for col in range(self._grid_width)] for row in range(self._grid_height)]
        self._pixels_unit_width = self._pixels_screen_size / grid_width
        self._pixels_unit_height = self._pixels_screen_size / grid_height
        self._pixels_border_percentage = snake_pixels_border_percentage
        self._pixels_border_offset_x = self._pixels_unit_width * self._pixels_border_percentage / 2
        self._pixels_border_offset_y = self._pixels_unit_height * self._pixels_border_percentage / 2
        self._pixels_unit_center_width = self._pixels_unit_width * (1 - self._pixels_border_percentage)
        self._pixels_unit_center_height = self._pixels_unit_height * (1 - self._pixels_border_percentage)
        pygame.display.flip()
        print("\nSnake initialization finished.")

    def _set_grid_fill(self, row, col, pygame_color):
        # Must subtract one to the coordinates due to the hidden border around the outside of the game window
        start_x = (col-1) * self._pixels_unit_width
        start_y = (row-1) * self._pixels_unit_height
        center_x = start_x + self._pixels_border_offset_x
        center_y = start_y + self._pixels_border_offset_y
        rectangle = pygame.Rect(center_x, center_y, self._pixels_unit_center_width, self._pixels_unit_center_height)
        pygame.draw.rect(self.game_screen, pygame_color, rectangle)

    def _set_cell_empty(self, row, col):
        self._grid[row][col] = self._grid_cell_empty
        self._set_grid_fill(row, col, self._color_empty)

    def _set_cell_snake_head(self, row, col):
        self._grid[row][col] = self._grid_cell_snake_head
        self._set_grid_fill(row, col, self._color_snake_head)

    def _set_cell_snake_body(self, row, col):
        self._grid[row][col] = self._grid_cell_snake_body
        self._set_grid_fill(row, col, self._color_snake_body)

    def _set_cell_apple(self, row, col):
        self._grid[row][col] = self._grid_cell_apple
        self._set_grid_fill(row, col, self._color_apple)

    def _place_starting_snake(self):
        start_row = int(self._grid_height / 2)
        start_col = int(self._grid_width / 2)
        self._set_cell_snake_head(start_row, start_col)
        self._snake_list.append([start_row, start_col])
        self._set_cell_snake_body(start_row, start_col-1)
        self._snake_list.append([start_row, start_col-1])
        self._set_cell_snake_body(start_row, start_col-2)
        self._snake_list.append([start_row, start_col-2])
        self._set_cell_snake_body(start_row, start_col-3)
        self._snake_list.append([start_row, start_col-3])

    def _routine_reset_game(self):
        self._snake_list = []
        self._snake_direction = self._direction_right
        self._snake_direction_next = self._snake_direction
        self._grid = [[0 for col in range(self._grid_width)] for row in range(self._grid_height)]
        background_rectangle = pygame.Rect(0, 0, self._pixels_screen_size, self._pixels_screen_size)
        pygame.draw.rect(self.game_screen, self._color_empty, background_rectangle)
        self._place_starting_snake()

    # Call after the new direction has been determined but before snake moves in the next determined location
    def _routine_apple(self, next_head_location_row, next_head_location_col):
        will_eat_apple = self._grid[next_head_location_row][next_head_location_col] == self._grid_cell_apple
        if will_eat_apple:
            self._snake_status = self._snake_status_eaten_apple
            # Add a body cell to the snake at the tail location
            tail_location = self._snake_list[-1]
            self._snake_list.append([tail_location[0], tail_location[1]])
            # No other logic is needed:
            # the new head pixels replace the apple pixels
            # the new head cell value replaces the grid cell value for the apple

    # Call after the new direction has been determined but before snake moves in the next determined location
    def _routine_collision_edges(self, next_head_location_row, next_head_location_col):
        # Must compensate for the grid's extra row and column
        will_collide_horizontal_edges = next_head_location_row > self._grid_visible_height or next_head_location_row < 1
        will_collide_vertical_edges = next_head_location_col > self._grid_visible_width or next_head_location_col < 1
        will_collide_edge = will_collide_horizontal_edges or will_collide_vertical_edges
        if will_collide_edge:
            self._snake_status = self._snake_status_hit_edge

    def _routine_snake_move(self, next_head_location_row, next_head_location_col):
        cell_next = self._snake_list[0].copy()
        # Move head
        self._snake_list[0][0] = next_head_location_row
        self._snake_list[0][1] = next_head_location_col
        self._set_cell_snake_head(self._snake_list[0][0], self._snake_list[0][1])
        # Move each body part
        body_length = len(self._snake_list)
        for i in range(1, body_length):
            cell_current = self._snake_list[i].copy()
            self._snake_list[i][0] = cell_next[0]
            self._snake_list[i][1] = cell_next[1]
            self._set_cell_snake_body(self._snake_list[i][0], self._snake_list[i][1])
            cell_next = cell_current
        # Clear the tail, but only if it is the tail, in the case that the new head is where the tail was
        if self._grid[cell_next[0]][cell_next[1]] == self._grid_cell_snake_body:
            self._set_cell_empty(cell_next[0], cell_next[1])

    def _routine_inputs(self):
        direction = None
        if self._events_key_down:
            if self._events_key_down[-1].unicode == "w":
                direction = self._direction_up
            elif self._events_key_down[-1].unicode == "a":
                direction = self._direction_left
            elif self._events_key_down[-1].unicode == "s":
                direction = self._direction_down
            elif self._events_key_down[-1].unicode == "d":
                direction = self._direction_right
            if direction is not None and self._snake_direction != (direction[0]*-1, direction[1]*-1):
                self._snake_direction_next = direction

    def _routine_move_snake_by_delay(self):
        # Movement with set delay
        self._time_since_last_movement_secs += self._game_frame_delay
        if self._time_since_last_movement_secs > self._time_movement_delay_secs:
            self._snake_direction = self._snake_direction_next
            next_head_location_row = self._snake_list[0][0] + self._snake_direction[0]
            next_head_location_col = self._snake_list[0][1] + self._snake_direction[1]
            self._routine_collision_edges(next_head_location_row, next_head_location_col)
            self._routine_apple(next_head_location_row, next_head_location_col)
            self._routine_snake_move(next_head_location_row, next_head_location_col)
            self._time_since_last_movement_secs = 0

    def start(self):
        self._routine_reset_game()
        while not self.game_quit:
            self.update()
            game_over = self._snake_status == self._snake_status_hit_edge or \
                self._snake_status == self._snake_status_hit_body
            if game_over:
                self._routine_reset_game()
            self._snake_status = self._snake_status_nothing
            self._routine_inputs()
            self._routine_move_snake_by_delay()

    def set_speed(self, speed_factor):
        self._time_movement_delay_secs /= max(0, speed_factor)
