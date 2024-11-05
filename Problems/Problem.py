import os
import requests
import gzip
import shutil
import pandas as pd
import numpy as np
from typing import Optional

class Problem:
    best_known_lengths = {"a280": 2579, "ali535": 202339, "att48": 10628, "att532": 27686, "bayg29": 1610,
                          "bays29": 2020, "berlin52": 7542, "bier127": 118282, "brazil58": 25395, "brd14051": 469385,
                          "brg180": 1950, "burma14": 3323, "ch130": 6110, "ch150": 6528, "d198": 15780, "d493": 35002,
                          "d657": 48912, "d1291": 50801, "d1655": 62128, "d2103": 80450, "d15112": 1573084,
                          "d18512": 645238, "dantzig42": 699, "dsj1000_EUC_2D": 18659688, "dsj1000_CEIL_2D": 18660188,
                          "eil51": 426, "eil76": 538, "eil101": 629, "fl417": 11861, "fl1400": 20127, "fl1577": 22249,
                          "fl3795": 28772, "fnl4461": 182566, "fri26": 937, "gil262": 2378, "gr17": 2085, "gr21": 2707,
                          "gr24": 1272, "gr48": 5046, "gr96": 55209, "gr120": 6942, "gr137": 69853, "gr202": 40160,
                          "gr229": 134602, "gr431": 171414, "gr666": 294358, "hk48": 11461, "kroA100": 21282,
                          "kroB100": 22141, "kroC100": 20749, "kroD100": 21294, "kroE100": 22068, "kroA150": 26524,
                          "kroB150": 26130, "kroA200": 29368, "kroB200": 29437, "lin105": 14379, "lin318": 42029,
                          "linhp318": 41345, "nrw1379": 56638, "p654": 34643, "pa561": 2763, "pcb442": 50778,
                          "pcb1173": 56892, "pcb3038": 137694, "pla7397": 23260728, "pla33810": 66048945,
                          "pla85900": 142382641, "pr76": 108159, "pr107": 44303, "pr124": 59030, "pr136": 96772,
                          "pr144": 58537, "pr152": 73682, "pr226": 80369, "pr264": 49135, "pr299": 48191,
                          "pr439": 107217, "pr1002": 259045, "pr2392": 378032, "rat99": 1211, "rat195": 2323,
                          "rat575": 6773, "rat783": 8806, "rd100": 7910, "rd400": 15281, "rl1304": 252948,
                          "rl1323": 270199, "rl1889": 316536, "rl5915": 565530, "rl5934": 556045, "rl11849": 923288,
                          "si175": 21407, "si535": 48450, "si1032": 92650, "st70": 675, "swiss42": 1273,
                          "ts225": 126643, "tsp225": 3916, "u159": 42080, "u574": 36905, "u724": 41910, "u1060": 224094,
                          "u1432": 152970, "u1817": 57201, "u2152": 64253, "u2319": 234256, "ulysses16": 6859,
                          "ulysses22": 7013, "usa13509": 19982859, "vm1084": 239297, "vm1748": 336556}

    def __init__(self, problem_name: str = 'berlin52', num_ants: int = 10, alpha: float = 1.0,
                 beta: float = 2.0, evaporation_rate: float = 0.5, pheromone_rate: float = 100.0):
        """
        Initialize a Problem instance with ACO-specific parameters.

        :param problem_name: Name of the TSP problem file.
        :param num_ants: Number of ants.
        :param alpha: Influence of pheromone in decisions.
        :param beta: Influence of heuristic (inverse of distance).
        :param evaporation_rate: Rate of pheromone evaporation.
        :param pheromone_rate: Initial pheromone deposit amount.
        """
        self.name = problem_name
        self.local_file = f"Problems/{problem_name}.tsp"
        self.nodes = self.load_nodes(problem_name)
        self.distances = self.compute_distance_matrix() if self.nodes is not None else None
        self.best_known_length = self.best_known_lengths.get(problem_name, 0)

        # ACO parameters
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_rate = pheromone_rate

    def load_nodes(self, problem_name: str) -> Optional[pd.DataFrame]:
        """
        Load nodes from a TSP problem file. Download and extract if needed.
        """
        if not os.path.exists(self.local_file):
            self.download_and_extract_problem(problem_name)

        nodes_data = []
        try:
            with open(self.local_file, 'r') as file:
                reading_nodes = False
                for line in file:
                    line = line.strip()
                    if line == "NODE_COORD_SECTION":
                        reading_nodes = True
                        continue
                    if line == "EOF":
                        break
                    if reading_nodes:
                        parts = line.split()
                        nodes_data.append({'name': int(parts[0]), 'x': float(parts[1]), 'y': float(parts[2])})
            return pd.DataFrame(nodes_data)
        except FileNotFoundError:
            print(f"File {self.local_file} not found.")
            return None

    def download_and_extract_problem(self, problem_name: str) -> None:
        """
        Download and extract the problem file if not available locally.
        """
        url = f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{problem_name}.tsp.gz"
        os.makedirs(os.path.dirname(self.local_file), exist_ok = True)

        print(f"Downloading {problem_name}...")
        response = requests.get(url)
        response.raise_for_status()

        with open(f"{self.local_file}.gz", 'wb') as f:
            f.write(response.content)

        with gzip.open(f"{self.local_file}.gz", 'rb') as f_in, open(self.local_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(f"{self.local_file}.gz")

    def compute_distance_matrix(self) -> np.ndarray:
        """
        Calculate the Euclidean distance matrix for the nodes.
        """
        x, y = np.array(self.nodes['x']), np.array(self.nodes['y'])
        x_diffs, y_diffs = np.subtract.outer(x, x), np.subtract.outer(y, y)
        return np.sqrt(x_diffs ** 2 + y_diffs ** 2)

    def get_length(self, route: np.ndarray) -> float:
        """
        Calculate the length of a given route.

        :param route: Array of node indices representing the route.
        :return: Total length of the route.
        """
        if len(route) == 0 or self.distances is None:
            return 0.0

        total_length = np.sum(self.distances[route[:-1], route[1:]])
        total_length += self.distances[route[-1], route[0]]
        return total_length
