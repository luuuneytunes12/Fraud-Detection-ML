import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/creditcard_cleaned.csv")
days_covered = (df["Time"].max() - df["Time"].min()) / 86400
monthly_fraud = len(df[df['Class'] == 1]) * 30 / days_covered
monthly_txns = len(df) * 30 / days_covered

models = ["Dummy", "LightGBM", "PyTorch MLP", "Stacking"]
recall = [0.00, 0.80, 0.81, 0.78]
precision = [0.00, 0.70, 0.68, 0.87]

# Calculate F2 score (weights recall 2x more than precision)
f2 = [(5 * p * r) / (4 * p + r) if (p + r) > 0 else 0 
      for p, r in zip(precision, recall)]

tp = [monthly_fraud * r for r in recall]
fp = [t/p - t if p > 0 else 0 for t, p in zip(tp, precision)]
fp_rate = [f/monthly_txns * 100 for f in fp]
fraud_caught = [r*100 for r in recall]
f2_pct = [f*100 for f in f2]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Chart 1: Detection Rate
bars1 = ax1.bar(models, fraud_caught)
ax1.set_ylabel("% Fraud Caught")
ax1.set_title("Detection Rate (Recall)")
for bar, val in zip(bars1, fraud_caught):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{val:.0f}%', ha='center')

# Chart 2: F2 Score
bars2 = ax2.bar(models, f2_pct)
ax2.set_ylabel("F2 Score (%)")
ax2.set_title("F2 Score (Recall-Weighted)")
for bar, val in zip(bars2, f2_pct):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{val:.0f}%', ha='center')

# Chart 3: False Positive Rate
bars3 = ax3.bar(models, fp_rate)
ax3.set_ylabel("% False Alarms")
ax3.set_title("False Positive Rate")
for bar, val in zip(bars3, fp_rate):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{val:.3f}%', ha='center')

plt.tight_layout()
plt.show()

print(f"Stacking: {recall[3]*100:.0f}% fraud caught, F2={f2[3]:.2f}, {fp_rate[3]:.3f}% FP rate")
