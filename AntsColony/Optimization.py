from typing import Callable, Any
from AntsColony.Colony import Colony
from Problems import Problem

class Optimization:
    def __init__(self, problem: Problem, num_iterations: int = 100):
        """
        Initialize the Optimization object to manage the ACO process over multiple iterations.

        :param problem: A Problem instance containing nodes, distance matrix, and ACO parameters.
        :param num_iterations: Number of iterations for the optimization process.
        """
        self.num_iterations = num_iterations
        self.colony = Colony(problem)
        self.best_route = None
        self.best_length = float('inf')
        self.iteration_history = []

    def optimize(self, callback: Callable[[Any], None] = None) -> None:
        """
        Execute the optimization process, iterating through the ACO algorithm, updating the best solution,
        and calling a callback function (if provided) with current optimization status at each iteration.

        :param callback: Optional function called after each iteration with the current state.
        """
        for iteration in range(1, self.num_iterations + 1):
            # Run a single iteration and get the best route and length found in this iteration
            best_route, best_length = self.colony.run_iteration()

            # Update the overall best route if the current one is shorter
            if best_length < self.best_length:
                self.best_length = best_length
                self.best_route = best_route[:]

            # Append current iteration data to history
            self.iteration_history.append({
                'iteration': iteration,
                'best_route': self.best_route,
                'best_length': self.best_length
            })

            # Trigger the callback with current state information if provided
            if callback:
                callback({
                    'iteration': iteration,
                    'best_route': self.best_route,
                    'best_length': self.best_length,
                    'iteration_history': self.iteration_history
                })
