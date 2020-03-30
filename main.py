from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.properties import ObjectProperty

from kivy.config import Config
import numpy
Config.set('kivy', 'keyboard_mode', 'systemanddock')

def difference_with_y(y, fitness):
    return numpy.abs(fitness - y)

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.min(fitness)) # CHANGED MAX TO MIN!!!
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, amount_to_mutate=None, range_of_mutation=None):
    # Mutation changes a single gene in each offspring randomly.

    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-range_of_mutation, range_of_mutation, 1)
        for j in range(amount_to_mutate):
            offspring_crossover[idx, j] = offspring_crossover[idx, j] + random_value # CHANGED 4 TO 3!!!
    return offspring_crossover

def compute_ga(y, x1, x2, x3, x4):
    # Inputs of the equation.
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # equation_inputs = [4, 2,
    #                   5, -1]
    # y = 10
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    equation_inputs = [x1, x2, x3, x4]

    # Number of the weights we are looking to optimize.
    num_weights = 4

    """
    Genetic algorithm parameters:
    Mating pool size
    Population size
    """
    sol_per_pop = 8
    num_parents_mating = 4

    # Defining the population size.
    pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    #Creating the initial population.
    new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
    print(new_population)

    num_generations = 30
    params = {'amount_to_mutate' :[1,2,3],
              'range_of_mutation':[1.0,2.0,3.0,4.0]}
    fastest_params = [] # tuple(best_amount_to_mutate, range_of_mutation) of best from the best
    previous_best  = None # variable to compare with current best, if diff < 0.002 stop iteration
    for amount_to_mutate in params['amount_to_mutate']:
        for range_of_mutation in params['range_of_mutation']:
            for generation in range(num_generations):
                # Measing the fitness of each chromosome in the population.
                fitness = cal_pop_fitness(equation_inputs, new_population)
                fitness = difference_with_y(y, fitness)
                # Selecting the best parents in the population for mating.
                parents = select_mating_pool(new_population, fitness, 
                                                      num_parents_mating)

                # Generating next generation using crossover.
                offspring_crossover = crossover(parents,
                                               offspring_size=(pop_size[0]-parents.shape[0], num_weights))

                # Adding some variations to the offsrping using mutation.
                offspring_mutation = mutation(offspring_crossover,
                                              amount_to_mutate=amount_to_mutate,
                                              range_of_mutation=range_of_mutation)

                # Creating the new population based on the parents and offspring.
                new_population[0:parents.shape[0], :] = parents
                new_population[parents.shape[0]:, :] = offspring_mutation
                
                # The best result in the current iteration.
                result_current  = numpy.max(numpy.sum(new_population*equation_inputs, axis=1))
                if generation != 0:
                    result_current  = numpy.max(numpy.sum(new_population*equation_inputs, axis=1))
                    if numpy.abs(y - result_current) < 0.1:
                        best_match_idx = numpy.where(fitness == numpy.min(fitness))
                        current_best = new_population[best_match_idx, :][0][0]
                        fastest_params.append([generation,
                                               result_current,
                                               current_best, 
                                               amount_to_mutate,    # 3
                                               range_of_mutation])  # 4
                        break
                


    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    fitness = cal_pop_fitness(equation_inputs, new_population)
    fitness = difference_with_y(y, fitness)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness == numpy.min(fitness)) # CHANGED MAX TO MIN!!!
    least_generations = 30
    best_idx = 0
    for idx, params in enumerate(fastest_params):
        if fastest_params[idx][0] < least_generations:
            best_idx = idx
    print("Best solution : ", new_population[best_match_idx, :][0][0])
    print("Best solution fitness : ", fitness[best_match_idx])
    print("Best params for mutations: ", fastest_params[best_idx])
    print(f"Best amount to mutate:{fastest_params[best_idx][3]}, range for mutation:{fastest_params[best_idx][4]}") # <------ ADDITIONAL TASK OUTPUT
    return new_population[best_match_idx, :][0][0]    

class Container(GridLayout):
  def calculate(self):
    try:
      y   = float(self.y_val.text)
      x_1 = float(self.x_1.text)
      x_2 = float(self.x_2.text)
      x_3 = float(self.x_3.text)
      x_4 = float(self.x_4.text)
    except:
      y, x_1, x_2, x_3, x_4 = 0
		
    weights = compute_ga(y, *[x_1, x_2, x_3, x_4])
    w_1, w_2, w_3, w_4 = weights
    w_1, w_2, w_3, w_4 = round(w_1, 2), round(w_2, 2), round(w_3, 2), round(w_4, 2)
    answer = f"{y} = {x_1}*{w_1}+{x_2}*{w_2}+{x_3}*{w_3}+{x_4}*{w_4}"
    self.answer.text = answer


class MyApp(App):
  def build(self):
    return Container()

if __name__=='__main__':
  MyApp().run()
