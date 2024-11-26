'''
Contains the Selection class for selecting parents.
'''
class Selection:
    @staticmethod
    def select_parents(population, fitness_scores):
        return population[0], population[1]
