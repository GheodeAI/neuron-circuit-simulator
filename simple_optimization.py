import argparse
import numpy as np
import metaheuristic_designer as mhd
from simulator_fn import *
from fitness_fn_simple import *


def neuron_operator_volt(neuron_matrix, _1, _2):
    # print(neuron_matrix)
    random_matrix = generate_random_neuron_matrix(None)[:9]
    mask = np.zeros(9, dtype=bool)
    mask[:2] = 1
    np.random.shuffle(mask)
    neuron_matrix[mask] = random_matrix[mask]
    return neuron_matrix

def main(optimization_time=3600*10, simulation_time=100, fitness_repetitions=1, population_size=50, execution_idx="0"):
    objfunc = NeuronCircuitFit(simulation_time=simulation_time, repetitions=fitness_repetitions)

    encoding = NeuronAdjMatrixSimple()
    initializer = NeuronAdjMatrixSimpleInitializer(population_size, activation_p=0.6, encoding=encoding)
    # mutation = mhd.operators.VectorOperator("RandomMask", {"N": 4})
    mutation = mhd.operators.MetaOperator(
        "Split",
        [
            mhd.OperatorFromLambda(neuron_operator_volt, vectorized=False),
            mhd.operators.VectorOperator("MutSample", {"distrib": "Uniform", "low": 1, "up": 6, "N": 2}, encoding=mhd.encodings.TypeCastEncoding(int, float))
        ],
        params={"mask": np.concatenate([np.zeros(9), np.ones(9)]),}
    )
    # mutation = mhd.OperatorFromLambda(neuron_operator, vectorized=False)
    crossover = mhd.operators.VectorOperator("Multipoint")
    parent_sel_op = mhd.selection_methods.NullParentSelection()
    survivor_sel_op = mhd.selection_methods.SurvivorSelection("(m+n)")

    search_strategy = mhd.strategies.ES(initializer, mutation, crossover, parent_sel_op, survivor_sel_op, {"offspringSize":population_size*3})
    alg = mhd.algorithms.GeneralAlgorithm(
        objfunc,
        search_strategy,
        {"stop_cond": "time_limit", "time_limit": optimization_time, "verbose": True, "v_timer": 0.5}
    )

    try:
        result = alg.optimize()
    except KeyboardInterrupt:
        result = alg.search_strategy.population
        print("Interrupted execution.")

    best_solution, best_fitness = result.best_solution(decoded=True)
    np.save(f"./solutions/best_solution-{execution_idx}.npy", best_solution)
    np.save(f"./solutions/population-{execution_idx}.npy", result.genotype_matrix)

    with open(f"./solutions/best_fitness-{execution_idx}.txt", "w") as f:
        f.write(str(best_fitness))

    with open(f"./solutions/population_fitness-{execution_idx}.txt", "w") as f:
        f.write(", ".join(map(str, result.fitness * -1)))
    
    # alg.display_report(show_plots=True, save_fig=True, fig_name=f"./solutions/fitness_history-{execution_idx}.png")
    alg.display_report(show_plots=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--optim_time", dest="optim_time", help="Specify an algorithm", default=3600*10, type=float)
    parser.add_argument("-s", "--simtime", dest="simtime", help="Specify an algorithm", default=100, type=int)
    parser.add_argument("-r", "--repetitions", dest="repetitions", help="Specify an algorithm", default=5, type=int)
    parser.add_argument("-p", "--population_size", dest="population_size", help="Specify an algorithm", default=50, type=int)
    parser.add_argument("-x", "--execution_idx", dest="execution_idx", help="Specify an algorithm", default="0")
    args = parser.parse_args()

    main(optimization_time=args.optim_time, simulation_time=args.simtime, fitness_repetitions=args.repetitions, population_size=args.population_size, execution_idx=args.execution_idx)
