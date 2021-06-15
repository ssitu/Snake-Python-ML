import pygame
import time


class Game:
    def _update_game_events(self):
        self._events_quit = pygame.event.get(pygame.QUIT)
        self._events_key_down = pygame.event.get(pygame.KEYDOWN)
        self._events_key_up = pygame.event.get(pygame.KEYUP)

    def set_fps(self, fps):
        self._game_frame_delay = 1 / fps

    def __init__(self, width, height):
        pygame.init()
        self.game_screen = pygame.display.set_mode(size=(width, height))
        self.game_quit = False
        self.set_fps(60)
        self._update_game_events()
        self._external_routines_update = {}
        self._external_routines_quit = {}
        self._external_routines_key_down = {}
        self._external_routines_key_up = {}
        self._time_no_sleep = False

    def set_max_speed(self):
        self._game_frame_delay = 0
        self._time_no_sleep = True

    def add_routine_update(self, routine, parameters):
        self._external_routines_update[routine] = parameters

    def add_routine_quit(self, routine, parameters):
        self._external_routines_quit[routine] = parameters

    def add_routine_key_down(self, routine, parameters):
        self._external_routines_key_down[routine] = parameters

    def add_routine_key_up(self, routine, parameters):
        self._external_routines_key_up[routine] = parameters

    def _toggleable_sleep(self, seconds):
        if not self._time_no_sleep:
            time.sleep(self._game_frame_delay)

    @staticmethod
    def _call_dict_routines(dict_routines):
        for routine, parameters in dict_routines.items():
            routine(*parameters)

    def _routine_update(self):
        self._call_dict_routines(self._external_routines_update)

    def _routine_quit(self):
        self.game_quit = True
        self._call_dict_routines(self._external_routines_quit)

    def _routine_key_down(self):
        self._call_dict_routines(self._external_routines_key_down)

    def _routine_key_up(self):
        self._call_dict_routines(self._external_routines_key_up)

    def update(self):
        self._routine_update()
        self._update_game_events()
        if self._events_quit:
            self._routine_quit()
        if self._events_key_down:
            self._routine_key_down()
        if self._events_key_up:
            self._routine_key_up()
        pygame.display.flip()
        self._toggleable_sleep(self._game_frame_delay)
