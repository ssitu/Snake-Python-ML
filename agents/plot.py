import multiprocessing
import time
from multiprocessing.managers import BaseManager

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.axes import Axes
from matplotlib.figure import Figure

style.use("fivethirtyeight")


class ContinuousTask(multiprocessing.Process):
    def __init__(self, task, task_args=None):
        multiprocessing.Process.__init__(self)
        self.stop_event = multiprocessing.Event()
        self.pause_event = multiprocessing.Event()
        self.task = task
        self.task_args = task_args
        if task_args is None:
            self.task_args = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            while not self.pause_event.is_set():
                try:
                    self.task(*self.task_args)
                except (EOFError, BrokenPipeError):
                    raise Exception("Error in plotting process, the main thread may have ended abruptly.")

    def pause(self):
        self.pause_event.set()

    def unpause(self):
        self.pause_event.clear()

    def finish_up(self):
        self.pause()
        self.stop_event.set()


class PlotManager(BaseManager):

    def __init__(self, title, label_x, label_y, moving_avg_len=10, update_interval=0.01):
        super(PlotManager, self).__init__()
        self.start()
        self.plot: Plot = Plot(title, label_x, label_y,
                               moving_average_length=moving_avg_len, update_interval=update_interval)

    def plot_data(self, y: float):
        self.plot.plot_data(y)

    def draw_plot(self):
        self.plot.draw_plot()

    def start_updates(self):
        self.plot.start_updates()

    def pause_updates(self):
        self.plot.pause_updates()

    def unpause_updates(self):
        self.plot.unpause_updates()

    def stop_updates(self):
        self.plot.stop_updates()

    def save_plot(self, path):
        self.plot.save(path)

    def set_parameters(self, update_interval, moving_average_length):
        self.plot.set_parameters(update_interval, moving_average_length)

    def reset(self):
        self.plot.reset()


class Plot:
    __figure: Figure
    __ax: Axes

    def __init__(self, title, x_label, y_label, moving_average_length=10, update_interval=0.01):
        self.__title = title
        self.__x_label = x_label
        self.__y_label = y_label
        self.__moving_average_length = moving_average_length
        self.__moving_index = 0
        self.__moving_averages = multiprocessing.Manager().list()
        self.__moving_sum = 0
        self.__y_data = multiprocessing.Manager().list()
        self.__interval = update_interval
        self.__figure, self.__ax = plt.subplots(1, 1)
        self.__plot_loop = ContinuousTask(self.draw_plot)

    def plot_data(self, y: float):
        # Update the plot data
        self.__y_data.append(y)
        self.__moving_sum += y
        if len(self.__y_data) > self.__moving_average_length:
            # Moving window at the max
            self.__moving_sum -= self.__y_data[self.__moving_index]
            self.__moving_index += 1
            moving_average = self.__moving_sum / self.__moving_average_length
        else:
            # Moving window can be expanded
            moving_average = self.__moving_sum / len(self.__y_data)
        self.__moving_averages.append(moving_average)

    def draw_plot(self, show=True):
        if len(self.__y_data) > 0:
            self.__ax.cla()
            self.__ax.set_title(self.__title)
            self.__ax.set_xlabel(self.__x_label)
            self.__ax.set_ylabel(self.__y_label)
            y_data = list(self.__y_data)
            moving_data = list(self.__moving_averages)
            self.__ax.plot(y_data)
            self.__ax.plot(moving_data)
            self.__ax.text(len(y_data) - 1, y_data[-1], str(y_data[-1]))
            self.__ax.text(len(moving_data) - 1, moving_data[-1], str(moving_data[-1]))
            if show:
                plt.pause(self.__interval)

    def start_updates(self):
        self.__plot_loop.start()

    def pause_updates(self):
        self.__plot_loop.pause()

    def unpause_updates(self):
        self.__plot_loop.unpause()

    def stop_updates(self):
        self.__plot_loop.finish_up()
        time.sleep(1)  # Allow time for the plot process to shut down

    def save(self, path):
        self.draw_plot(show=False)  # Update the figure to the current data
        self.__figure.savefig(path, bbox_inches="tight")

    def set_parameters(self, update_interval, moving_average_length):
        self.__interval = update_interval
        self.__moving_average_length = moving_average_length

    def reset(self):
        self.__y_data.clear()
        self.__moving_averages.clear()
        self.__moving_sum = 0
        self.__moving_index = 0
