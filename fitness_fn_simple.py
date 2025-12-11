import numpy as np
import metaheuristic_designer as mhd
# from simulator_fn import run_simulator, generate_random_neuron_matrix
from simulator_fn import run_simulator, generate_random_neuron_matrix, calculate_frequency, count_peaks


class NeuronCircuitFit(mhd.VectorObjectiveFunc):
    def __init__(self, simulation_time=100, neuron_amounts=(0, 0, 0, 0, 0, 12, 6, 2), bin_size=0.2, target_freq=0.17, repetitions=3, d=1.0):
        self.simulation_time = simulation_time
        self.neuron_amounts = neuron_amounts
        self.n_neurons = sum(neuron_amounts)
        self.target_freq = target_freq
        self.repetitions = repetitions
        self.bin_size = bin_size
        self.d = d
        self.alpha = 0.001
        self.beta = 0.001
        self.gamma = 10
        low_lim = np.concatenate([np.repeat(-18, 9), np.repeat(0, 9)])
        up_lim = np.concatenate([np.repeat(18, 9), np.repeat(6, 9)])
        super().__init__(vecsize=18, low_lim=low_lim, up_lim=up_lim, mode="min", name="Fit neural circuit", vectorized=False)

    def objective(self, solution):
        volt_mat = solution[0]
        conn_mat = solution[1]

        analysis_start = 1000 # starting point from which to calculate the statistics of the frequency graph, in ms

        hmean_freq = 0
        hmean_naive_freq = 0
        mean_lowact = 0
        mean_n_events = 0
        for _ in range(self.repetitions):
            sim = run_simulator(
                volt_mat, conn_matrix=conn_mat, simulation_time=self.simulation_time, neuron_amounts=self.neuron_amounts, record_volt=False
            )

            freq_graph = calculate_frequency(sim, simulation_time=self.simulation_time, bin_size=self.bin_size)
            mean_n_events += count_peaks(freq_graph, d=self.d)

            n_low_activtions = 0
            for activations in sim:
                n_low_activtions += int(len(activations) < 3)
            mean_lowact += n_low_activtions/len(sim)

        mean_n_events /= self.repetitions
        if mean_n_events == 0:
            mean_freq = np.inf
        else:
            mean_freq = mean_n_events / self.simulation_time

        freq_target = (mean_freq - self.target_freq) ** 2

        ## Isolated subgraphs
        adjmat = (volt_mat != 0) & (conn_mat != 0)
        adjmat = adjmat[-3:, -3:]

        # A bit of a hack to calculate the number of isolated subgraphs. Ignore connections from a node to itself.
        # In bigger graphs you could do BFS or calculate (I - A)^{n-1} and check that every non-diagonal entry is 1.
        # There are 2 subgraphs if no nodes have at least 2 edges. There are 3 if there are no edges.
        # Convert the directed graph to an undirected graph, then perform the algorithm.
        v01 = int(adjmat[0,1] | adjmat[1,0])
        v12 = int(adjmat[1,2] | adjmat[2,1])
        v02 = int(adjmat[0,2] | adjmat[2,0])

        n_isolated_subgraphs = 3 - ((v01 | v12) + (v02 | v12) + (v01 | v02)) + (v01 | v12 | v02)
        return freq_target + self.alpha * mean_lowact + self.beta * (n_isolated_subgraphs - 1)

    def repair_solution(self, solution):
        return solution

    def repair_speed(self, speed):
        return speed


class NeuronAdjMatrixSimple(mhd.Encoding):
    def __init__(self):
        super().__init__(vectorized=True, decode_as_array=True)

    def encode_func(self, solution):
        return solution[:, :, -3:, -3:].flatten()

    def decode_func(self, solution):
        # print(solution[:, :9])
        # print(solution[:, 9:])
        # print(np.any(solution[:, 9:] == 0))
        solution = solution.copy()
        n_solutions = solution.shape[0]
        result_matrix = np.zeros((n_solutions, 2, 8, 8))

        volt_mat = solution[:, :9].reshape((-1, 3, 3))
        n_conn = solution[:, 9:].reshape((-1, 3, 3))

        volt_mat[n_conn == 0] = 0
        n_conn[volt_mat == 0] = 0

        result_matrix[:, 0, -3:, -3:] = volt_mat
        result_matrix[:, 1, -3:, -3:] = n_conn

        return result_matrix


class NeuronAdjMatrixSimpleInitializer(mhd.Initializer):
    inhibition = np.array([-8, -13, -18])
    excitation = np.array([8, 13])
    connection_strengths = {
        "ISe": np.hstack([inhibition, excitation]),
        "ISi": np.hstack([inhibition, excitation]),
        "IFB": inhibition,
        "ISB": inhibition,
        "A": np.array([18]),
        "RS": inhibition,
        "RSB": excitation,
        "RFB": excitation,
    }

    def __init__(self, pop_size, activation_p=0.5, encoding=None):
        self.activation_p = activation_p
        super().__init__(pop_size=pop_size, encoding=encoding)

    def generate_random(self):
        # return generate_random_neuron_matrix(self.activation_p)
        return generate_random_neuron_matrix(self.activation_p)
