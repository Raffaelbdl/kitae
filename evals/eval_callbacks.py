import os
from datetime import datetime
from time import time

from rl.callback import Callback, CallbackData


class TimeCallback(Callback):
    def __init__(self, task: str, n_envs: int) -> None:
        self.task = task
        self.n_envs = n_envs

        self.update_duration_list = []
        self.start_train_time = 0.0
        self.start_update_time = 0.0

        self.training_duration = 0.0
        self.update_duration = 0.0

    def write_results(self, evaluated: str):
        filepath = os.path.join(
            os.path.dirname(__file__),
            evaluated,
            datetime.now().strftime("%Y_%m_%d_%H_%M"),
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.writelines(
                [
                    "TimeCallback Outputs >>>>> \n",
                    f"Task : {self.task} | n_envs : {self.n_envs} \n",
                    f"Training Duration : {self.training_duration:.1f} \n",
                    f"Mean Update Duration : {self.update_duration:.1f} \n",
                ]
            )

    def on_train_start(self, callback_data: CallbackData):
        self.start_train_time = time()

    def on_update_start(self, callback_data: CallbackData):
        self.start_update_time = time()

    def on_update_end(self, callback_data: CallbackData):
        self.update_duration_list.append(time() - self.start_update_time)

    def on_train_end(self, callback_data: CallbackData):
        self.training_duration = time() - self.start_train_time

        self.update_duration = sum(self.update_duration_list)
        self.update_duration /= len(self.update_duration_list)

        print("Training duration : ", self.training_duration)
        print("Mean update duration : ", self.update_duration)
