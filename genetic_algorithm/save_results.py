import pandas as pd
import matplotlib.pyplot as plt


def save_results_to_csv(weights, assets, output_file):
    """
    Save portfolio weights to a CSV file.

    Args:
        weights (np.ndarray): Optimal portfolio weights.
        assets (list): List of asset names.
        output_file (str): Path to save the CSV file.
    """
    results = pd.DataFrame({"Asset": assets, "Weight": weights})
    results.to_csv(output_file, index=False)
    print(f"Portfolio weights saved to {output_file}")


def save_pie_chart(weights, assets, output_file, title="Optimized Portfolio Allocation"):
    """
    Save a pie chart of the portfolio weights as a PNG.

    Args:
        weights (np.ndarray): Optimal portfolio weights.
        assets (list): List of asset names.
        output_file (str): Path to save the PNG file.
        title (str): Title of the pie chart.
    """
    # Filter out assets with zero weights
    filtered_assets = [asset for asset, weight in zip(assets, weights) if weight > 0]
    filtered_weights = [weight for weight in weights if weight > 0]

    if not filtered_weights:
        print("No assets with weights greater than 0 to plot.")
        return

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(filtered_weights, labels=filtered_assets, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.savefig(output_file, format="png")
    print(f"Portfolio allocation chart saved to {output_file}")
