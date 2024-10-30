'''
Contains the Portfolio class to represent each portfolio (chromosome).
'''
class Portfolio:
    def __init__(self, weights):
        self.weights = weights

    @staticmethod
    def random_portfolio(num_assets):
        return Portfolio([0] * num_assets)
