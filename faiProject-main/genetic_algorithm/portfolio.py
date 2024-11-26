import numpy as np
'''
Contains the Portfolio class to represent each portfolio (chromosome).
'''
class Portfolio:
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.normalize()

    @staticmethod

    
    def random_portfolio(num_assets):
        """
        make random portfolio.

        """
        weights = np.random.rand(num_assets)
        weights = weights / np.sum(weights)
        return Portfolio(weights)
    
    def normalize(self):
        '''
        Makes sure the weights equal 1
        '''
        total = np.sum(self.weights)
        self.weights = self.weights / total
