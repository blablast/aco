import numpy as np
from typing import List, Optional

class Pheromone:
    def __init__(self, num_nodes: int, evaporation_rate: float, pheromone_deposit: float):
        """
        Initialize the Pheromone matrix for Ant Colony Optimization.

        :param num_nodes: Total number of nodes in the graph.
        :param evaporation_rate: Percentage of pheromone to evaporate per iteration (0.0 to 1.0).
        :param pheromone_deposit: Base amount of pheromone to deposit per ant.
        """
        self.trails = np.full((num_nodes, num_nodes), 0.1, dtype=float)  # Initial pheromone values
        self.evaporation_rate = np.clip(evaporation_rate, 0.0, 1.0)  # Limit evaporation rate between 0 and 1
        self.pheromone_deposit = pheromone_deposit

    def evaporate(self, evaporation_rate: Optional[float] = None) -> None:
        """
        Evaporate pheromones on all trails by the specified rate.

        :param evaporation_rate: Optional new evaporation rate to apply.
        """
        self.evaporation_rate = np.clip(evaporation_rate, 0.0, 1.0) if evaporation_rate is not None else self.evaporation_rate
        self.trails *= (1 - self.evaporation_rate)

    def deposit(self, route: List[int], length: float, pheromone_deposit: Optional[float] = None) -> None:
        """
        Add pheromone to trails based on an ant's route.

        :param route: List of node indices representing the ant's path.
        :param length: Total length of the ant's route, inversely affecting deposit.
        :param pheromone_deposit: Optional pheromone deposit amount, overwriting the default if provided.
        """
        deposit_amount = (pheromone_deposit or self.pheromone_deposit) / length

        # Deposit pheromone on each edge in the route
        for i, j in zip(route[:-1], route[1:]):
            self.trails[i, j] += deposit_amount
            self.trails[j, i] += deposit_amount

        # Add pheromone for the last-to-first connection if the route is circular
        if route[0] != route[-1]:
            i, j = route[-1], route[0]
            self.trails[i, j] += deposit_amount
            self.trails[j, i] += deposit_amount
