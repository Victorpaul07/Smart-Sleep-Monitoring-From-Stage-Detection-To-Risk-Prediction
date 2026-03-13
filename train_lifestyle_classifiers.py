import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_recall_fscore_support

# ===============================
# CONFIG
# ===============================
DATA_PATH = "data/sleep_health_lifestyle_dataset.csv"
MODEL_DIR = "models"
PLOTS_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Training HYBRID Sleep Disorder Predictor (RF+SVM)...")

# ===============================
# LOAD & PREPROCESS
# ===============================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

bp_col = 'Blood Pressure (systolic/diastolic)'
df[bp_col] = df[bp_col].astype(str).str.split('/').str[0].astype(float)
df['Blood Pressure'] = df[bp_col]
df = df.drop(bp_col, axis=1)

feature_cols = [
    'Gender', 'Age', 'Occupation',
    'Sleep Duration (hours)', 'Quality of Sleep (scale: 1-10)',
    'Physical Activity Level (minutes/day)', 'Stress Level (scale: 1-10)',
    'BMI Category', 'Blood Pressure', 'Heart Rate (bpm)', 'Daily Steps'
]

X = df[feature_cols].copy()
y = df['Sleep Disorder']

# Encode
le_features = {}
for col in ['Gender', 'Occupation', 'BMI Category']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_features[col] = le

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

scaler = StandardScaler()
num_cols = ['Age', 'Sleep Duration (hours)', 'Quality of Sleep (scale: 1-10)',
            'Physical Activity Level (minutes/day)', 'Stress Level (scale: 1-10)',
            'Blood Pressure', 'Heart Rate (bpm)', 'Daily Steps']
X[num_cols] = scaler.fit_transform(X[num_cols])

print(f"Classes: {np.bincount(y_encoded)}")

# ===============================
# SMOTE OVERSAMPLING
# ===============================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X, y_encoded)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"After SMOTE - Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Balanced train: {np.bincount(y_train)}")

# ===============================
# TRAIN MODELS & TRACK METRICS
# ===============================
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_loss = 1 - rf_acc  # Error rate as "loss"

print("\nTraining SVM...")
svm_model = SVC(
    kernel='rbf', C=10, gamma='scale', probability=True,
    class_weight='balanced', random_state=42
)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_loss = 1 - svm_acc

print("\nTraining HYBRID Model (RF+SVM)...")
hybrid_model = VotingClassifier(
    estimators=[('rf', rf_model), ('svm', svm_model)],
    voting='soft'
)
hybrid_model.fit(X_train, y_train)
hybrid_pred = hybrid_model.predict(X_test)
hybrid_acc = accuracy_score(y_test, hybrid_pred)
hybrid_loss = 1 - hybrid_acc

# Print Results
print("\n" + "="*50)
print("MODEL COMPARISON RESULTS")
print("="*50)
print(f"Random Forest     - Accuracy: {rf_acc:.4f}, Loss: {rf_loss:.4f}")
print(f"SVM              - Accuracy: {svm_acc:.4f}, Loss: {svm_loss:.4f}")
print(f"Hybrid (RF+SVM)  - Accuracy: {hybrid_acc:.4f}, Loss: {hybrid_loss:.4f}")
print("="*50)

print("\nHybrid Model Classification Report:")
print(classification_report(y_test, hybrid_pred, target_names=le_target.classes_, zero_division=0))

# ===============================
# SAVE MODELS
# ===============================
dump(hybrid_model, os.path.join(MODEL_DIR, 'sleep_disorder_hybrid.joblib'))
dump(rf_model, os.path.join(MODEL_DIR, 'sleep_disorder_rf.joblib'))
dump(le_features, os.path.join(MODEL_DIR, 'lifestyle_encoders.joblib'))
dump(le_target, os.path.join(MODEL_DIR, 'disorder_encoder.joblib'))
dump(scaler, os.path.join(MODEL_DIR, 'lifestyle_scaler.joblib'))

# ===============================
# PLOT 1: ACC AND LOSS COMPARISON (Like Reference Image)
# ===============================
fig, ax = plt.subplots(figsize=(10, 7))

models = ['RF', 'SVM', 'Hybrid\n(RF+SVM)']
acc_values = [rf_acc, svm_acc, hybrid_acc]
loss_values = [rf_loss, svm_loss, hybrid_loss]

x = np.arange(len(models))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, acc_values, width, label='Accuracy', color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, loss_values, width, label='Loss', color='#95a5a6', edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Styling
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Model Accuracy and Loss Comparison\nSleep Disorder Prediction', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_acc_loss.png'), dpi=300, bbox_inches='tight')
print(f"\nAccuracy comparison plot saved: {PLOTS_DIR}/model_comparison_acc_loss.png")

# ===============================
# PLOT 2: COMPREHENSIVE ANALYSIS
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 2.1 Confusion Matrix
cm = confusion_matrix(y_test, hybrid_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_,
            cbar_kws={'label': 'Count'})
axes[0, 0].set_title('Hybrid Model Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Predicted', fontsize=12)
axes[0, 0].set_ylabel('Actual', fontsize=12)

# 2.2 Feature Importance
importances = rf_model.feature_importances_
imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
axes[0, 1].barh(imp_df['feature'], imp_df['importance'], color=colors, edgecolor='black')
axes[0, 1].set_title('Feature Importance (Random Forest Component)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Importance', fontsize=12)

# 2.3 Model Accuracy Comparison
models_full = ['Random\nForest', 'SVM', 'Hybrid\n(RF+SVM)']
accuracies = [rf_acc, svm_acc, hybrid_acc]
colors_acc = ['#3498db', '#e74c3c', '#27ae60']
bars = axes[1, 0].bar(models_full, accuracies, color=colors_acc, edgecolor='black', linewidth=2)
axes[1, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylim([0, 1.0])
axes[1, 0].axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% Baseline')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Add accuracy values on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2.4 Prediction Confidence Distribution
hybrid_proba = hybrid_model.predict_proba(X_test)
conf = np.max(hybrid_proba, axis=1)
axes[1, 1].hist(conf, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(conf.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {conf.mean():.3f}')
axes[1, 1].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Max Probability', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
print(f"Comprehensive analysis plot saved: {PLOTS_DIR}/comprehensive_analysis.png")

plt.show()

# ===============================
# PLOT 3: Precision / Recall / F1 (separate plot per model)
# Uses weighted average for multi-class imbalance
# ===============================
def get_prf(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return float(p), float(r), float(f1)

metrics_map = {
    "Random Forest": get_prf(y_test, rf_pred),
    "SVM": get_prf(y_test, svm_pred),
    "Hybrid (RF+SVM)": get_prf(y_test, hybrid_pred)
}

for model_name, (p, r, f1) in metrics_map.items():
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Precision", "Recall", "F1-score"]
    values = [p, r, f1]
    colors = ["#3498db", "#f39c12", "#27ae60"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.5)

    # Value labels on top
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_title(f"{model_name} - Precision/Recall/F1 (Weighted Avg)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = os.path.join(PLOTS_DIR, f"prf_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PRF plot saved: {out_path}")

print("TRAINING COMPLETE!")
print(f"Models saved in: {MODEL_DIR}/")
print(f"Plots saved in: {PLOTS_DIR}/")
print(f"Best Model: sleep_disorder_hybrid.joblib")
print(f"Final Accuracy: {hybrid_acc:.2%}")
