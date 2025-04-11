
import numpy as np
import pandas as pd

# reproducibility
np.random.seed(42)


# Define model parameters

NUM_CUSTOMERS = 3000
NUM_DAYS = 30

STATES = ['Good', 'At-Risk', 'Delinquent', 'Default', 'Closed']
STATE_INDEX = {state: i for i, state in enumerate(STATES)}

# Transition matrix (5x5) 
TRANSITION_MATRIX = np.array([
    [0.90, 0.07, 0.01, 0.00, 0.02],  # Good
    [0.10, 0.70, 0.15, 0.02, 0.03],  # At-Risk
    [0.00, 0.10, 0.65, 0.20, 0.05],  # Delinquent
    [0.00, 0.00, 0.00, 1.00, 0.00],  # Default (absorbing)
    [0.00, 0.00, 0.00, 0.00, 1.00],  # Closed (absorbing)
])


# Generate initial balances
# ----------------------------
def generate_balances(num_customers):
    # Log-normal distribution 
    return np.random.lognormal(mean=6.0, sigma=0.5, size=num_customers)

# Simulate transitions

def simulate_customers(num_customers, num_days):
    balances = generate_balances(num_customers)
    records = []

    for customer_id in range(1, num_customers + 1):
        current_state = 'Good'
        balance = balances[customer_id - 1]

        for day in range(num_days):
            records.append({
                "CustomerID": customer_id,
                "Day": day,
                "State": current_state,
                "Balance": round(balance, 2)
            })

            if current_state in ['Default', 'Closed']:
                break  # Stop simulating for this customer

            state_idx = STATE_INDEX[current_state]
            next_state = np.random.choice(STATES, p=TRANSITION_MATRIX[state_idx])
            current_state = next_state

    df = pd.DataFrame(records)
    return df


def save_data(df, path="data/simulated_data.csv"):
    df.to_csv(path, index=False)
    print(f" Simulated data saved to {path}")


if __name__ == "__main__":
    df_simulated = simulate_customers(NUM_CUSTOMERS, NUM_DAYS)
    save_data(df_simulated)
