# AVO Risk Model – Customer Financial Risk Simulation

This project models and visualizes customer financial risk using a Markov Decision Process (MDP) and Reinforcement Learning. Designed for AVO, a neobank focused on emerging markets, the model simulates transitions between customer states (e.g., Good, At-Risk, Default) and calculates expected financial losses over time. It includes data simulation, risk evaluation logic, rich visualizations, and extensions into machine learning and RL-based strategies.

---

## 📊 Project Components

### 1. Simulated Customer Behavior
- `simulate_data.py`: Simulates 3,000 customers over 30 days using a defined state transition matrix.
- States: Good → At-Risk → Delinquent → Default / Closed
- Balance data is generated using a log-normal distribution.

### 2. Loss Modeling
- `model_logic.py`: Calculates expected loss and cumulative loss based on absorbing states (Default = 100% loss, Closed = 50% loss).
- Aggregates risk per day for financial exposure analysis.

### 3. Visualizations
- `visualize.py`:
  - Line plot: Daily vs. cumulative expected loss
  - Stacked bar chart: State distribution over time

### 4. Reinforcement Learning Simulation
- `rl_env.py`: Implements a custom Gym environment simulating customer risk dynamics.
- Action space: Do nothing, Warn, or Freeze account
- Custom reward logic incentivizes timely intervention on risky customers
- Includes optional Tkinter-based pop-up viewer for RL logs

### 5. AI Prediction (Optional Extension)
- `ai_predictor.py`: Uses Random Forest to predict a customer's next state based on current behavior, balance, and day.

---

## 📁 Project Structure

```bash
avo-risk-model/
├── simulate_data.py        # Simulates and stores customer risk transitions
├── model_logic.py          # Calculates daily and cumulative expected losses
├── visualize.py            # Generates loss and state distribution charts
├── rl_env.py               # Custom Gym RL environment for intervention testing
├── ai_predictor.py         # ML-based next-state predictor (Random Forest)
├── data/                   # Contains CSVs, charts, and simulated outputs
└── README.md               # Project documentation

How to Run
Install dependencies: pip install pandas matplotlib seaborn numpy scikit-learn gym tqdm

Simulate customer data: python simulate_data.py

Calculate expected losses: python model_logic.py

Visualize losses and state distributions: python visualize.py

Run the RL environment with pop-up results: python rl_env.py

Predict next state using AI: python ai_predictor.py

📈 Outputs
simulated_data.csv: Synthetic customer transitions

loss_chart.png: Daily vs. cumulative loss

state_distribution.png: State changes over time

rl_run_log.txt: (Optional) Log of RL agent decisions and rewards

💡 Future Improvements
Add customer metadata (region, account type) for richer simulations

Integrate Q-learning into the RL environment for training agents

Deploy model outputs as dashboards using Streamlit or Dash

Add API layer to serve risk scores to downstream systems

🙌 Author
Kehinde Obidele
Health Informatics | Risk Modeling | Reinforcement Learning
