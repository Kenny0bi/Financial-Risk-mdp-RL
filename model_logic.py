
import pandas as pd


# Define loss per absorbing state
LOSS_RULES = {
    'Default': 1.00,  # 100% loss
    'Closed': 0.50    # 50% loss
}


def load_simulated_data(path="data/simulated_data.csv"):
    df = pd.read_csv(path)
    return df


# daily expected loss
def calculate_expected_loss(df):
    df = df.copy()
    
    # Add loss column only for absorbing states
    df['Loss'] = df.apply(
        lambda row: row['Balance'] * LOSS_RULES.get(row['State'], 0.0),
        axis=1
    )

    # Filter only the first occurrence of each absorbing state per customer
    absorbing_df = df[df['State'].isin(LOSS_RULES.keys())]
    first_absorptions = absorbing_df.drop_duplicates(subset='CustomerID', keep='first')

    # Daily expected loss
    daily_loss = first_absorptions.groupby('Day')['Loss'].sum().reset_index()
    daily_loss['CumulativeLoss'] = daily_loss['Loss'].cumsum()

    return daily_loss


if __name__ == "__main__":
    df = load_simulated_data()
    loss_df = calculate_expected_loss(df)

    print("Daily Expected Loss:")
    print(loss_df.head(10))  
