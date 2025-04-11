import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv("data/simulated_data.csv")

# Encode states numerically
state_encoder = LabelEncoder()
df['StateEncoded'] = state_encoder.fit_transform(df['State'])

# Create "NextState" column (shifted)
df['NextState'] = df.groupby('CustomerID')['StateEncoded'].shift(-1)
df = df.dropna()

# Features: current state, balance, day
X = df[['Day', 'Balance', 'StateEncoded']]
y = df['NextState']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=state_encoder.classes_)

print("State Prediction Model Evaluation:")
print(report)
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

def show_popup_report(report_text):
    root = tk.Tk()
    root.title("AI State Prediction Report")

    text_widget = ScrolledText(root, wrap=tk.WORD, width=80, height=20)
    text_widget.insert(tk.END, report_text)
    text_widget.pack(padx=10, pady=10)
    
    root.mainloop()

show_popup_report(report)

