import random
import copy
import torch
from training import trainingfcn

def evaluate_candidate(candidate, train_tensor, test_tensor, eps, lr, batch_size, S_p, T, M):
    """
    Evaluates a candidate by running a shortened training (e.g. using fewer epochs)
    and returns the test loss.
    """
    alpha = [candidate['alpha0'], candidate['alpha1'], candidate['alpha2']]
    try:
        # trainingfcn returns a tuple; the second element is the test loss.
        results = trainingfcn(eps, lr, batch_size, S_p, T, alpha,
                              candidate['Num_meas'], candidate['Num_inputs'],
                              candidate['Num_x_Obsv'], candidate['Num_x_Neurons'],
                              candidate['Num_u_Obsv'], candidate['Num_u_Neurons'],
                              candidate['Num_hidden_x_encoder'], candidate['Num_hidden_x_decoder'],
                              candidate['Num_hidden_u_encoder'], candidate['Num_hidden_u_decoder'],
                              train_tensor, test_tensor, M)
        test_loss = results[1]
    except Exception as e:
        print("Error evaluating candidate:", candidate, e)
        test_loss = float('inf')
    return test_loss

def initialize_population(pop_size):
    """
    Create an initial population of candidate hyperparameter sets.
    """
    population = []
    for _ in range(pop_size):
        candidate = {
            "Num_meas": random.randint(1, 3),
            "Num_inputs": random.randint(1, 3),
            "Num_x_Obsv": random.randint(1, 5),
            "Num_u_Obsv": random.randint(1, 5),
            "Num_x_Neurons": random.randint(10, 50),
            "Num_u_Neurons": random.randint(10, 50),
            "Num_hidden_x_encoder": random.randint(1, 3),
            "Num_hidden_x_decoder": random.randint(1, 3),
            "Num_hidden_u_encoder": random.randint(1, 3),
            "Num_hidden_u_decoder": random.randint(1, 3),
            "alpha0": random.uniform(0.01, 1.0),
            "alpha1": random.uniform(1e-9, 1e-5),
            "alpha2": random.uniform(1e-18, 1e-12)
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

def mutate(candidate, mutation_rate=0.1):
    """
    With a given probability, randomly change each hyperparameter.
    Integer parameters are perturbed by ±1 (or ±5 for neurons) and floats are scaled.
    """
    if random.random() < mutation_rate:
        candidate['Num_meas'] = max(1, candidate['Num_meas'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['Num_inputs'] = max(1, candidate['Num_inputs'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['Num_x_Obsv'] = max(1, candidate['Num_x_Obsv'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['Num_u_Obsv'] = max(1, candidate['Num_u_Obsv'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['Num_x_Neurons'] = max(10, candidate['Num_x_Neurons'] + random.choice([-5, 5]))
    if random.random() < mutation_rate:
        candidate['Num_u_Neurons'] = max(10, candidate['Num_u_Neurons'] + random.choice([-5, 5]))
    if random.random() < mutation_rate:
        candidate['Num_hidden_x_encoder'] = max(1, candidate['Num_hidden_x_encoder'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['Num_hidden_x_decoder'] = max(1, candidate['Num_hidden_x_decoder'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['Num_hidden_u_encoder'] = max(1, candidate['Num_hidden_u_encoder'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['Num_hidden_u_decoder'] = max(1, candidate['Num_hidden_u_decoder'] + random.choice([-1, 1]))
    if random.random() < mutation_rate:
        candidate['alpha0'] = max(0.01, min(1.0, candidate['alpha0'] * random.uniform(0.8, 1.2)))
    if random.random() < mutation_rate:
        candidate['alpha1'] = max(1e-9, min(1e-5, candidate['alpha1'] * random.uniform(0.8, 1.2)))
    if random.random() < mutation_rate:
        candidate['alpha2'] = max(1e-18, min(1e-12, candidate['alpha2'] * random.uniform(0.8, 1.2)))
    return candidate

def run_genetic_algorithm(train_tensor, test_tensor, generations=5, pop_size=10, eps=50, lr=1e-3, batch_size=256, S_p=30, M=1):
    """
    Runs the genetic algorithm over a number of generations and returns the best candidate.
    
    Parameters:
      - train_tensor, test_tensor: the data tensors used for evaluation
      - generations: number of generations to run
      - pop_size: population size per generation
      - eps: number of epochs for evaluation training (use a small value here to speed up GA)
      - lr, batch_size, S_p, M: other training parameters (as in your trainingfcn)
    """
    T = train_tensor.shape[1]  # Assuming shape: (num_samples, T, features)
    population = initialize_population(pop_size)
    
    best_candidate = None
    best_fitness = -float('inf')  # Fitness = -loss, so higher fitness is better
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        fitnesses = []
        for candidate in population:
            loss = evaluate_candidate(candidate, train_tensor, test_tensor, eps, lr, batch_size, S_p, T, M)
            fitness = -loss  # Lower loss => higher fitness
            fitnesses.append(fitness)
            print(f"Candidate: {candidate} | Loss: {loss}")
            if fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate
        
        new_population = []
        # Create a new population via tournament selection, crossover, and mutation
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
        print(f"Best candidate in generation {gen+1}: {best_candidate} (Fitness: {-best_fitness})")
    
    print("Best candidate overall:", best_candidate)
    return best_candidate
