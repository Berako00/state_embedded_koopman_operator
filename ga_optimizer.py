import random
import copy
import torch
from training import trainingfcn, trainingfcn_mixed

def evaluate_candidate(check_epoch, breakout, candidate, train_tensor, test_tensor, eps, lr, batch_size, S_p, T, M, device=None):
    """
    Evaluates a candidate by running a shortened training using fewer epochs
    and returns the test loss.
    """
    alpha = [candidate['alpha0'], candidate['alpha1'], candidate['alpha2']]
    try:
        results = trainingfcn(eps, check_epoch, lr, batch_size, S_p, T, alpha, candidate['Num_meas'], candidate['Num_inputs'], candidate['Num_x_Obsv'], candidate['Num_x_Neurons'], candidate['Num_u_Obsv'], candidate['Num_u_Neurons'],
                                candidate['Num_hidden_x'], candidate['Num_hidden_u'], candidate['Num_hidden_u'], train_tensor, test_tensor, M, device=device)
        # Use only the lowest_loss (first element) for fitness evaluation
        lowest_loss = results[0]
    except Exception as e:
        print("Error evaluating candidate:", candidate, e)
        lowest_loss = float('inf')
    return lowest_loss


def initialize_population(pop_size, param_ranges, Num_meas, Num_inputs):
    """
    Create an initial population of candidate hyperparameter sets.
    """
    population = []
    for _ in range(pop_size):
        candidate = {
            "Num_meas": Num_meas,
            "Num_inputs": Num_inputs,
            "Num_x_Obsv": random.randint(*param_ranges["Num_x_Obsv"]),
            "Num_u_Obsv": random.randint(*param_ranges["Num_u_Obsv"]),
            "Num_x_Neurons": random.randint(*param_ranges["Num_x_Neurons"]),
            "Num_u_Neurons": random.randint(*param_ranges["Num_u_Neurons"]),
            "Num_hidden_x": random.randint(*param_ranges["Num_hidden_x"]),  # Shared x hidden layers
            "Num_hidden_u": random.randint(*param_ranges["Num_hidden_u"]),  # Shared u hidden layers
            "alpha0": random.uniform(*param_ranges["alpha0"]),
            "alpha1": random.uniform(*param_ranges["alpha1"]),
            "alpha2": random.uniform(*param_ranges["alpha2"])
        }
        population.append(candidate)
    return population


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Selects a candidate from the population using tournament selection.
    Here, fitness is defined as negative loss so that a lower loss is a higher fitness.
    """
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    # sort so that the best (largest fitness, i.e. smallest loss) comes first
    selected.sort(key=lambda x: x[1], reverse=True)
    return copy.deepcopy(selected[0][0])

def crossover(parent1, parent2):
    """
    Performs uniform crossover: for each hyperparameter, randomly choose a parent's value.
    """
    child = {}
    for key in parent1.keys():
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child

def mutate(candidate, param_ranges, mutation_rate=0.1):
    """
    With a given probability, randomly change each hyperparameter.
    Integer parameters are perturbed by ±1 (or ±5 for neurons) and floats are scaled.
    """
    if random.random() < mutation_rate:
        candidate['Num_x_Obsv'] = max(param_ranges["Num_x_Obsv"][0], min(param_ranges["Num_x_Obsv"][1], candidate['Num_x_Obsv'] + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        candidate['Num_u_Obsv'] = max(param_ranges["Num_u_Obsv"][0], min(param_ranges["Num_u_Obsv"][1], candidate['Num_u_Obsv'] + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        candidate['Num_x_Neurons'] = max(param_ranges["Num_x_Neurons"][0], min(param_ranges["Num_x_Neurons"][1], candidate['Num_x_Neurons'] + random.choice([-5, 5])))
    if random.random() < mutation_rate:
        candidate['Num_u_Neurons'] = max(param_ranges["Num_u_Neurons"][0], min(param_ranges["Num_u_Neurons"][1], candidate['Num_u_Neurons'] + random.choice([-5, 5])))
    if random.random() < mutation_rate:
        candidate['Num_hidden_x'] = max(param_ranges["Num_hidden_x"][0], min(param_ranges["Num_hidden_x"][1], candidate['Num_hidden_x'] + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        candidate['Num_hidden_u'] = max(param_ranges["Num_hidden_u"][0], min(param_ranges["Num_hidden_u"][1], candidate['Num_hidden_u'] + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        new_alpha0 = candidate['alpha0'] * (10 ** random.choice([-1, 1]))
        candidate['alpha0'] = max(param_ranges["alpha0"][0], min(param_ranges["alpha0"][1], new_alpha0))
    if random.random() < mutation_rate:
        new_alpha1 = candidate['alpha1'] * (10 ** random.choice([-2, -1, 1, 2]))
        candidate['alpha1'] = max(param_ranges["alpha1"][0], min(param_ranges["alpha1"][1], new_alpha1))
    if random.random() < mutation_rate:
        new_alpha2 = candidate['alpha2'] * (10 ** random.choice([-2, -1, 1, 2]))
        candidate['alpha2'] = max(param_ranges["alpha2"][0], min(param_ranges["alpha2"][1], new_alpha2))
    return candidate


def run_genetic_algorithm(check_epoch, breakout, Num_meas, Num_inputs, train_tensor, test_tensor, tournament_size, mutation_rate, generations=5, pop_size=10, eps=50, lr=1e-3, batch_size=256, S_p=30, M=1, param_ranges=None, elitism_count=1, device=None):
    """
    Runs the genetic algorithm over a number of generations and returns the best candidate.

    Parameters:
      - train_tensor, test_tensor: the data tensors used for evaluation
      - generations: number of generations to run
      - pop_size: population size per generation
      - eps: number of epochs for evaluation training (use a small value here to speed up GA)
      - lr, batch_size, S_p, M: other training parameters (as in your trainingfcn)
      - param_ranges: dictionary containing ranges for hyperparameters
      - elitism_count: number of top candidates to carry over to the next generation unchanged
    """
    if param_ranges is None:
        raise ValueError("Parameter ranges must be provided.")

    T = train_tensor.shape[1]  # Assuming shape: (num_samples, T, features)
    population = initialize_population(pop_size, param_ranges, Num_meas, Num_inputs)

    best_candidate = None
    best_fitness = -float('inf')  # Fitness = -loss, so higher fitness is better


    for gen in range(generations):
        # For generations after the first, store the best candidate from the previous generation
        if gen > 0:
            prev_best_candidate = best_candidate
            prev_best_fitness = best_fitness

        print(f"Generation {gen+1}/{generations}")
        fitnesses = []
        # Evaluate fitness for each candidate
        for candidate in population:
           # Skip evaluation if candidate is the best from the previous generation
            if gen > 0 and candidate == prev_best_candidate:
                fitness = prev_best_fitness
                loss = -fitness  # Since fitness = -loss
                print(f"Candidate (best from previous generation): {candidate} | Loss: {loss} (evaluation skipped)")

            else:
              loss = evaluate_candidate(check_epoch, breakout, candidate, train_tensor, test_tensor, eps, lr, batch_size, S_p, T, M, device)
              fitness = -loss  # Lower loss => higher fitness
              print(f"Candidate: {candidate} | Loss: {loss}")

            fitnesses.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate

        # Sort population by fitness (highest first)
        sorted_population = [cand for cand, fit in sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)]
        # Elitism: carry over top candidates unchanged
        elite_candidates = [copy.deepcopy(ind) for ind in sorted_population[:elitism_count]]

        new_population = elite_candidates.copy()

        # Create the rest of the new population via tournament selection, crossover, and mutation
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size=tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size=tournament_size)
            child = crossover(parent1, parent2)
            child = mutate(child, param_ranges, mutation_rate=mutation_rate)
            # Optional: ensure child is not identical to parent1
            while parent1 == child:
                child = mutate(child, param_ranges)
            new_population.append(child)

        population = new_population
        print(f"Best candidate in generation {gen+1}: {best_candidate} (Loss: {-best_fitness})")

    print("Best candidate overall:", best_candidate)
    return best_candidate
