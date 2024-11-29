from ga_loadData import load_data
from genetic_algorithm import GeneticAlgorithm
from save_results import save_results_to_csv, save_pie_chart


def main():
    # File paths
    mean_returns_file = "genetic_algorithm/processed_data/mean_returns.csv"
    covariance_file = "genetic_algorithm/processed_data/covariance_matrix.csv"
    daily_returns_file = "genetic_algorithm/processed_data/daily_returns.csv"
    results_csv = "genetic_algorithm/processed_data/optimized_portfolio.csv"
    results_png = "genetic_algorithm/processed_data/portfolio_allocation.png"

    # Load data
    mean_returns, covariance_matrix, daily_returns = load_data(mean_returns_file, covariance_file, daily_returns_file)

    # Run Genetic Algorithm
    ga = GeneticAlgorithm(mean_returns, covariance_matrix, population_size=100, generations=500, mutation_rate=0.02)
    optimal_weights = ga.evolve()

    # Save results
    save_results_to_csv(optimal_weights, mean_returns.index.tolist(), results_csv)
    save_pie_chart(optimal_weights, mean_returns.index.tolist(), results_png)


if __name__ == "__main__":
    main()
