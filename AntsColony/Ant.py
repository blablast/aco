import numpy as np
from typing import List, Set

class Ant:
    def __init__(self, start_node: int, num_nodes: int, alpha: float, beta: float):
        """
        Initialize an Ant for the Ant Colony Optimization algorithm.

        :param start_node: The starting node for the ant's journey.
        :param num_nodes: The total number of nodes in the graph.
        :param alpha: Influence of pheromone levels on node selection.
        :param beta: Influence of heuristic information (inverse of distance) on node selection.
        """
        self.start_node: int = start_node
        self.route: List[int] = [self.start_node]
        self.visited: Set[int] = {self.start_node}
        self.total_distance: float = 0.0
        self.num_nodes: int = num_nodes
        self.alpha: float = alpha
        self.beta: float = beta

    def reset(self) -> None:
        """Reset the ant's journey, keeping the start node as the starting point."""
        self.route = [self.start_node]
        self.visited = {self.start_node}
        self.total_distance = 0.0

    def find_route(self, distances: np.ndarray, pheromones: np.ndarray) -> None:
        """
        Construct a complete route by visiting all nodes.

        :param distances: Matrix of distances between nodes.
        :param pheromones: Matrix of pheromone levels between nodes.
        """
        self.reset()  # Reset the ant to its initial state

        # Visit each node once until all nodes are covered
        while len(self.route) < self.num_nodes:
            next_node = self._choose_next_node(distances, pheromones)
            current_node = self.route[-1]
            self.route.append(next_node)
            self.visited.add(next_node)
            self.total_distance += distances[current_node, next_node]

        # Return to the starting node to complete the route
        self.route.append(self.start_node)
        self.total_distance += distances[self.route[-2], self.start_node]

    def _choose_next_node(self, distances: np.ndarray, pheromones: np.ndarray) -> int:
        """
        Select the next node to visit based on pheromone and heuristic values.

        :param distances: Matrix of distances between nodes.
        :param pheromones: Matrix of pheromone levels between nodes.
        :return: Index of the next node to visit.
        """
        current_node = self.route[-1]
        available_nodes = [node for node in range(self.num_nodes) if node not in self.visited]

        # Get pheromone and distance values for available nodes
        pheromone_levels = pheromones[current_node, available_nodes]
        distances_to_nodes = distances[current_node, available_nodes]

        # Calculate heuristic values (1 / distance) raised to beta
        heuristic_values = np.where(distances_to_nodes > 0, (1.0 / distances_to_nodes) ** self.beta, 0)

        # Calculate selection probabilities
        probabilities = pheromone_levels ** self.alpha * heuristic_values

        # If all probabilities are zero, select the nearest available node
        if probabilities.sum() == 0:
            return available_nodes[np.argmin(distances_to_nodes)]

        probabilities /= probabilities.sum()  # Normalize to form a probability distribution
        return np.random.choice(available_nodes, p=probabilities)
