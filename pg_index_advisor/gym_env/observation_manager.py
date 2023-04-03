import logging

import numpy as np

from gym import spaces


VERY_HIGH_BUDGET = 100_000_000_000


class ObservationManager(object):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions

        self.number_of_features = 0

    def _init_episode(self, episode_state):
        self.episode_budget = episode_state["budget"]
        if self.episode_budget is None:
            self.episode_budget = VERY_HIGH_BUDGET

        self.initial_cost = episode_state["initial_cost"]

    def init_episode(self, episode_state):
        raise NotImplementedError

    def get_observation(self, environment_state):
        raise NotImplementedError

    def get_observation_space(self):
        observation_space = spaces.Box(
            low=self._create_low_boundaries(),
            high=self._create_high_boundaries(),
            shape=self._create_shape()
        )

        logging.info(f"Creating ObservationSpace with {self.number_of_features} features.")

        return observation_space

    def _create_shape(self):
        return (self.number_of_features,)

    def _create_low_boundaries(self):
        low = [-np.inf for feature in range(self.number_of_features)]

        return np.array(low)

    def _create_high_boundaries(self):
        high = [np.inf for feature in range(self.number_of_features)]

        return np.array(high)


class MultiColumnObservationManager(ObservationManager):
    def __init__(self, number_of_actions, config):
        ObservationManager.__init__(self, number_of_actions)

        self.number_of_query_classes = config["number_of_query_classes"]

        self.number_of_features = (
            self.number_of_actions  # Indicates for each action whether it was taken or not
            + self.number_of_query_classes  # The frequencies for every query class
            + 1  # The episode's budget
            + 1  # The current storage consumption
            + 1  # The initial workload cost
            + 1  # The current workload cost
        )
        self.frequencies = self._init_frequencies()

    def init_episode(self, episode_state):
        episode_workload = episode_state["workload"]
        super()._init_episode(episode_state)
        self.frequencies = np.array(self._get_frequencies_from_workload_wide(episode_workload))

    def get_observation(self, environment_state):
        observation = np.array(environment_state["action_status"])
        # TODO: maybe use extend? frequencies is an array
        observation = np.append(observation, self.frequencies)
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation

    def _get_frequencies_from_workload_wide(self, workload):
        frequencies = self._init_frequencies()

        for query in workload.queries:
            # query numbers stat at 1
            frequencies[query.nr - 1] = query.frequency

        return frequencies

    def _init_frequencies(self):
        return [0 for query in range(self.number_of_query_classes)]
