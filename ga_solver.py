from colorama import Fore, Style, init
import numpy as np
from tqdm import tqdm
import time

init(autoreset=True)


def ga_solver(returns, n=100, D=None, CR=0.9, F=0.8, itermax=1000,
              alpha=1, beta=1, gamma=0.1, delta=0.1, lambda_range=(0, 1)):

    if D is None:
        D = returns.shape[1]

    V = np.cov(returns, rowvar=False)
    r = np.mean(returns, axis=0)

    def objective_function(x, lamb):
        return lamb * np.dot(r, x) - (1 - lamb) * np.dot(np.dot(x, V), x)

    def penalty_function(x):
        return alpha * np.sum(np.abs(np.minimum(x, 0)) - np.minimum(x, 0)) \
            - beta * (np.abs(np.sum(x) - 1) - (np.sum(x) - 1))

    def fitness_function(x, lamb):
        obj_func = objective_function(x, lamb)
        pen_func = penalty_function(x)
        return obj_func + pen_func

    def create_initial_population():
        pop = np.random.rand(n, D)
        pop /= np.sum(pop, axis=1).reshape(-1, 1)
        return pop

    def mutate(pop):
        new_pop = np.empty_like(pop)
        for i in range(n):
            candidates = np.delete(pop, i, axis=0)
            idx1, idx2 = np.random.choice(
                candidates.shape[0], 2, replace=False)
            new_pop[i] = pop[i] + F * (candidates[idx1] - candidates[idx2])
        return new_pop

    def crossover(pop, new_pop):
        for i in range(n):
            crossover_mask = np.random.rand(D) < CR
            new_pop[i, crossover_mask] = pop[i, crossover_mask]

            new_pop[i] = np.maximum(new_pop[i], 0)
            new_pop[i] /= np.sum(new_pop[i])

        return new_pop

    def select(pop, new_pop, lamb):
        max_fitness = float('-inf')
        for i in range(n):
            fitness_pop_i = fitness_function(pop[i], lamb)
            fitness_new_pop_i = fitness_function(new_pop[i], lamb)

            if np.less(fitness_pop_i, fitness_new_pop_i):
                pop[i] = new_pop[i]

            max_fitness = max(max_fitness, fitness_pop_i, fitness_new_pop_i)
        return pop, max_fitness

    def ga_iteration(pop, lamb):
        new_pop = mutate(pop)
        new_pop = crossover(pop, new_pop)
        pop, max_fitness = select(pop, new_pop, lamb)
        return pop, max_fitness

    def optimize_portfolio(lamb):
        pop = create_initial_population()
        max_fitness = float('-inf')
        start_time = time.time()
        with tqdm(range(itermax), desc='Optimizing portfolios', colour='GREEN') as progress_bar:
            for _ in progress_bar:
                pop, iter_max_fitness = ga_iteration(pop, lamb)
                max_fitness = max(max_fitness, iter_max_fitness)

                elapsed_time = time.time() - start_time
                progress_bar.set_description(
                    f"Max fitness: {max_fitness:.4f}, Elapsed time: {elapsed_time:.2f}s")

        best_idx = np.argmin([fitness_function(x, lamb) for x in pop])
        return pop[best_idx]

    lambda_values = np.linspace(lambda_range[0], lambda_range[1], num=10)
    portfolios = [optimize_portfolio(lamb) for lamb in lambda_values]

    return portfolios
