import numpy as np


TOTAL_STAFF = 20
STAFF_HOURS = 8
SHIFTING_DAYS = 5
WAGES_WEIGHT = 0.5
STAFF_WEIGHT = 0.3
DAYS_WEIGHT = 0.2

# Objective function coefficients
C1 = 10  # Replace with your actual coefficient for staff hours
C2 = 5   # Replace with your actual coefficient for shifting days
C3 = 8   # Replace with your actual coefficient for wages
# Function to initialize the population
def initialize_population(population_size, chromosome_length):
    return np.random.randint(2, size=(population_size, chromosome_length))

# Function to calculate the objective function value (cost)
# ... (previous code)


# ... (rest of the code remains the same)
def fitness_function(chromosome):
   # Decode chromosome into variables (adjust as per your encoding scheme)
   total_staff_assigned = np.sum(chromosome)
   total_staff_hours = total_staff_assigned * STAFF_HOURS
   total_shifting_days = np.sum(chromosome.reshape((TOTAL_STAFF, -1)), axis=0)

   # Calculate cost using the objective function formula
   cost = C1 * total_staff_hours + C2 * np.sum(total_shifting_days) + C3 * np.sum(chromosome * WAGES_WEIGHT)

   return cost


# Fitness function based on the inverse of the objective function (minimization problem)
def tournament_selection(chromosomes, fitness, elite_count):
 selected_indices = []

 if elite_count >= len(chromosomes):
     # Handle the case where elite_count is equal to or greater than the population size
     return list(range(len(chromosomes)))

 for _ in range(len(chromosomes) - elite_count):
     tournament_indices = np.random.choice(len(chromosomes), size=2, replace=False)
     selected_index = tournament_indices[np.argmax(fitness[tournament_indices])]
     selected_indices.append(selected_index)

 return selected_indices
  
# Function for crossover
def crossover(parent1, parent2, crossover_rate):
   if np.random.rand() < crossover_rate:
       # Two-point crossover
       crossover_points = np.sort(np.random.choice(len(parent1), size=2, replace=False))
       child1 = np.concatenate((parent1[:crossover_points[0]], parent2[crossover_points[0]:crossover_points[1]], parent1[crossover_points[1]:]))
       child2 = np.concatenate((parent2[:crossover_points[0]], parent1[crossover_points[0]:crossover_points[1]], parent2[crossover_points[1]:]))
       return child1, child2
   else:
       return parent1, parent2

# Function for mutation
def mutate(chromosome, mutation_rate):
    mutation_point = np.random.randint(len(chromosome))
    if np.random.rand() < mutation_rate:
        chromosome[mutation_point] = 1 - chromosome[mutation_point]  # Flip the bit
    return chromosome

# Genetic Algorithm
def genetic_algorithm(population_size, chromosome_length, elite_count, crossover_rate, mutation_rate, max_generations, stall_generation):
   population = initialize_population(population_size, chromosome_length)
   fitness = np.array([fitness_function(chromosome) for chromosome in population])
   best_fitness = np.max(fitness)
   best_solution = population[np.argmax(fitness)]
   generations_without_improvement = 0

   for generation in range(max_generations):
       # Elitism: Carry the best individuals to the next generation
       elite_indices = np.argsort(fitness)[-elite_count:]
       new_population = population[elite_indices].tolist()

       # Tournament selection
       selected_indices = tournament_selection(population, fitness, elite_count)

       # Create the rest of the new population through crossover and mutation
       while len(new_population) < population_size:
           parent1, parent2 = [population[i] for i in np.random.choice(selected_indices, 2, replace=False)]
           child1, child2 = crossover(parent1, parent2, crossover_rate)
           child1 = mutate(child1, mutation_rate)
           child2 = mutate(child2, mutation_rate)
           new_population.extend([child1, child2])

       # Replace the current population with the new population
       population = np.array(new_population[:population_size])
       fitness = np.array([fitness_function(chromosome) for chromosome in population])

       # Check for improvement in fitness
       current_best_fitness = np.max(fitness)
       if current_best_fitness > best_fitness:
           best_fitness = current_best_fitness
           best_solution = population[np.argmax(fitness)]
           generations_without_improvement = 0
       else:
           generations_without_improvement += 1

       # Check for stall generation
       if generations_without_improvement >= stall_generation:
           print("Algorithm converged. Stopping early.")
           break

   # Return the best solution found
   return best_solution


# Example usage
elite_count = 1
population_size = 100
chromosome_length = 840
crossover_rate = 0.8
mutation_rate = 0.002
max_generations = 500
stall_generation = 100

best_solution = genetic_algorithm(population_size, chromosome_length, elite_count, crossover_rate, mutation_rate, max_generations, stall_generation)

# Print or use the best solution as needed
print("Best Solution:", best_solution)
print("Objective Function Value:", objective_function(best_solution))
print("Fitness Value:", fitness_function(best_solution))
