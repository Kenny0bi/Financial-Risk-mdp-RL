import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model_logic import calculate_expected_loss, load_simulated_data

sns.set(style="whitegrid")


df = load_simulated_data()
loss_df = calculate_expected_loss(df)

# Plot 1: Daily Expected Loss
def plot_expected_loss():
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['Day'], loss_df['Loss'], marker='o', label='Daily Expected Loss')
    plt.plot(loss_df['Day'], loss_df['CumulativeLoss'], linestyle='--', label='Cumulative Loss')
    plt.xlabel("Day")
    plt.ylabel("USD")
    plt.title("Expected vs Cumulative Loss Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/loss_chart.png")
    plt.show()


# Plot 2: Customer State Distribution Over Time
def plot_state_distribution():
    state_counts = df.groupby(['Day', 'State']).size().unstack(fill_value=0)
    state_counts.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='tab20')
    plt.xlabel("Day")
    plt.ylabel("Number of Customers")
    plt.title("Customer States Over Time")
    plt.tight_layout()
    plt.savefig("data/state_distribution.png")
    plt.show()


if __name__ == "__main__":
    plot_expected_loss()
    plot_state_distribution()
