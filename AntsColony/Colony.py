from typing import List, Tuple
import random
import time
from AntsColony.Ant import Ant
from AntsColony.Pheromone import Pheromone
from Problems.Problem import Problem


class Colony:
    def __init__(self, problem: Problem):
        """
        Initialize the Colony object for the Ant Colony Optimization algorithm.

        :param problem: A Problem instance with nodes, distance matrix, and ACO parameters.
        """
        self.problem = problem
        self.num_nodes = len(self.problem.nodes)
        self.pheromones = Pheromone(self.num_nodes, problem.evaporation_rate, problem.pheromone_rate)

        # Initialize ants with parameters from the problem instance
        self.ants = [
            Ant(random.randint(0, self.num_nodes - 1), self.num_nodes, problem.alpha, problem.beta)
            for _ in range(problem.num_ants)
        ]

        # Best route tracking
        self.best_route = list(range(self.num_nodes))
        self.best_length = float('inf')
        self.best_iteration = 0
        self.best_iteration_time_ms = 0

        # Timing and iteration tracking
        self.iteration_history = []
        self.total_time_ms = 0
        self.current_iteration = 0

    def run_iteration(self) -> Tuple[List[int], float]:
        """
        Run a single iteration of the ACO algorithm, updating the best route and length if found.

        :return: Tuple containing the best route and its length.
        """
        self._prepare_iteration()
        self._perform_iteration()
        self._finalize_iteration()
        return self.best_route, self.best_length

    def _prepare_iteration(self) -> None:
        """Initialize timing and increment iteration count for each new iteration."""
        self.current_iteration += 1
        self._start_time = time.time()

    def _perform_iteration(self) -> None:
        """Run the main logic of the iteration: route finding, pheromone updating, and ant resetting."""
        for ant in self.ants:
            # Each ant finds a route based on pheromone trails and the distance matrix
            ant.find_route(self.problem.distances, self.pheromones.trails)
            route_length = self.problem.get_length(ant.route)

            # Update best route information if the current route is shorter
            if route_length < self.best_length:
                self.best_length = route_length
                self.best_route = ant.route[:]
                self.best_iteration = self.current_iteration
                self.best_iteration_time_ms = self.total_time_ms

        self._update_pheromones()

        # Reset all ants for the next iteration
        for ant in self.ants:
            ant.reset()

    def _update_pheromones(self) -> None:
        """Apply evaporation to all trails and deposit pheromones from top-performing ants."""
        self.pheromones.evaporate()

        # Top 5 ants contribute to pheromone deposits
        top_ants = sorted(self.ants, key = lambda ant: ant.total_distance)[:5]
        for ant in top_ants:
            self.pheromones.deposit(ant.route, ant.total_distance)

    def _finalize_iteration(self) -> None:
        """Conclude iteration timing, update cumulative time, and store iteration details in history."""
        iteration_time_ms = (time.time() - self._start_time) * 1000  # Convert seconds to milliseconds
        self.total_time_ms += iteration_time_ms

        # Record iteration data in history
        self.iteration_history.append({
            'iteration': self.current_iteration,
            'iteration_time_ms': iteration_time_ms,
            'cumulative_time_ms': self.total_time_ms,
            'best_length': self.best_length
        })

    def debug(self):
        """Output colony and problem state for debugging."""
        print(f"Number of nodes: {self.num_nodes}")
        for node in self.problem.nodes:
            print(f"Node {node.idx}, name: {node.name}")
            for dest_idx, distance in node.distances.items():
                print(f"  To Node {dest_idx}: {distance} km")
