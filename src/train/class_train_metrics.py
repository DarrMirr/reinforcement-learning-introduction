from collections import deque
from util.logger import log


class TrainMetrics():
    """
    Class contains metrics of train process.

    Parameters:
        rate (int): Metrics recording frequency.
        max_len (int): Metrics storage size.

    Attributes:
        episode_metrics_deque (deque): Metrics related to particular episode.
        current_episode_metric (EpisodeMetrics): Metrics of current training episode.
        episodes_created (int): Total count of created episodes.

    Author:
        Darr Mirr (Vladimir S. Polukeev)
    """

    def __init__(self, rate: int, max_len=1000):
        assert rate > 0
        assert max_len > 0

        self.rate = rate
        self.episode_metrics_deque: deque["EpisodeMetrics"] = deque(
            [], maxlen=max_len)
        self.current_episode_metric: EpisodeMetrics = None
        self.episodes_created: int = 0

        log.info(f'Train metrics created : storage_size={max_len}, rate={self.rate}')

    def start_new_episode(self, new_episode_num: int):
        if new_episode_num not in self.episode_metrics_deque:
            episode_metrics = EpisodeMetrics(new_episode_num)
            self.current_episode_metric = episode_metrics
            self.episodes_created += 1

            if self.episodes_created % self.rate == 0:
                self.episode_metrics_deque.append(episode_metrics)

    def end_current_episode(self):
        self.current_episode_metric = None

    def episode_metrics(self) -> "EpisodeMetrics":
        return self.current_episode_metric

    def to_list_of_dictionaries(self) -> list[dict]:
        list = []
        for epoch_metrics in self.episode_metrics_deque:
            dict = epoch_metrics.to_dictionary()
            list.append(dict)

        return list




class EpisodeMetrics():

    def __init__(self, episode_num: int):
        self.episode_num: int = episode_num
        self.exploration_action_num: int = 0
        self.exploration_action_positive: int = 0
        self.explotation_action_num: int = 0
        self.explotation_action_positive: int = 0
        self.episode_step_num: int = 0
        self.is_successfully_terminated: bool = False
        self.loss_fn_output : list[float] = []


    def inc_exploration_action_num(self, is_positive_exp: bool = False):
        self.exploration_action_num += 1
        if (is_positive_exp):
            self.exploration_action_positive += 1


    def inc_explotation_action_num(self, is_positive_exp: bool = False):
        self.explotation_action_num += 1
        if (is_positive_exp):
            self.explotation_action_positive += 1


    def inc_episode_step_num(self):
        self.episode_step_num += 1


    def mark_episode_as_successfully_terminated(self):
        self.is_successfully_terminated = True


    def to_dictionary(self) -> dict:
        return vars(self)


    def append_loss_fn_output(self, value: float):
        if value is not None:
            self.loss_fn_output.append(value)
