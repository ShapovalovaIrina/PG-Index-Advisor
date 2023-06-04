import logging

import numpy as np

from gym import spaces
from pg_index_advisor.gym_env.embeddings.workload_embedder import PlanEmbedderLSI


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
            shape=self._create_shape(),
            dtype=np.float64
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


class EmbeddingObservationManager(ObservationManager):
    def __init__(self, number_of_actions, config):
        ObservationManager.__init__(self, number_of_actions)

        workload_embedder_config = config["workload_embedder"]
        self.workload_embedder = PlanEmbedderLSI(
            workload_embedder_config["query_texts"],
            workload_embedder_config["representation_size"],
            workload_embedder_config["globally_indexable_columns"],
            workload_embedder_config["db_config"]
        )

        self.representation_size = self.workload_embedder.representation_size
        self.workload_size = config["workload_size"]

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True

        self.number_of_features = (
            self.number_of_actions  # Indicates for each action whether it was taken or not
            + (
                self.representation_size * self.workload_size
            )  # embedding representation for every query in the workload
            + self.workload_size  # The frequencies for every query in the workloads
            + 1  # The episode's budget
            + 1  # The current storage consumption
            + 1  # The initial workload cost
            + 1  # The current workload cost
        )

        self.workload_embedding = None

    def _init_episode(self, episode_state):
        episode_workload = episode_state["workload"]
        self.frequencies = self._get_frequencies_from_workload(episode_workload)

        super()._init_episode(episode_state)

    def init_episode(self, episode_state):
        raise NotImplementedError

    def get_observation(self, environment_state):
        workload_embedding = self._get_embeddings_from_environment_state(environment_state)

        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, workload_embedding)
        observation = np.append(observation, self.frequencies)
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation

    @staticmethod
    def _get_frequencies_from_workload(workload):
        frequencies = []

        for query in workload.queries:
            frequencies.append(query.frequency)

        return np.array(frequencies)

    def _get_embeddings_from_environment_state(self, environment_state):
        def _get_embeddings_array():
            return np.array(self.workload_embedder.get_embeddings(
                environment_state["plans_per_query"]
            ))

        if self.UPDATE_EMBEDDING_PER_OBSERVATION:
            workload_embedding = _get_embeddings_array()
        else:
            # In this case the workload embedding is not updated with every step
            # but also not set during init
            if self.workload_embedding is None:
                self.workload_embedding = _get_embeddings_array()

            workload_embedding = self.workload_embedding

        return workload_embedding


class SingleColumnIndexPlanEmbeddingObservationManagerWithCost(EmbeddingObservationManager):
    def __init__(self, number_of_actions, config):
        super().__init__(number_of_actions, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True

        # This overwrites EmbeddingObservationManager's features
        self.number_of_features = (
            self.number_of_actions  # Indicates for each action whether it was taken or not
            + (
                self.representation_size * self.workload_size
            )  # embedding representation for every query in the workload
            + self.workload_size  # The costs for every query in the workload
            + self.workload_size  # The frequencies for every query in the workloads
            + 1  # The episode's budget
            + 1  # The current storage consumption
            + 1  # The initial workload cost
            + 1  # The current workload cost
        )

    def init_episode(self, episode_state):
        super()._init_episode(episode_state)

    def get_observation(self, environment_state):
        workload_embedding = self._get_embeddings_from_environment_state(environment_state)

        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, workload_embedding)
        observation = np.append(observation, environment_state["costs_per_query"])
        observation = np.append(observation, self.frequencies)
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation

