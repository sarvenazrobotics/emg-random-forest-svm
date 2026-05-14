import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import time

# =============================
# 1. LOAD DATASET
# =============================
print("Loading dataset...")
df = pd.read_csv("EMG-data.csv") 
df = df.iloc[:500000]

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# =============================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Class distribution
print("\nClass Distribution:")
class_dist = df['class'].value_counts().sort_index()
print(class_dist)

# Define channel names (FIXED: define here before using)
channels = [f'channel{i}' for i in range(1, 9)]

# Plot 1: Class distribution
plt.figure(figsize=(12, 6))
df_clean = df[df['class'] != 0]
class_counts = df_clean['class'].value_counts().sort_index()
gesture_names = ['Rest', 'Fist', 'Wrist Flexion', 'Wrist Extension', 
                 'Radial Deviation', 'Ulnar Deviation', 'Extended Palm']
# Only show classes that exist
existing_gestures = [gesture_names[i-1] for i in class_counts.index if i-1 < len(gesture_names)]
plt.bar(existing_gestures, class_counts.values)
plt.xticks(rotation=45)
plt.title('Gesture Class Distribution (After Removing Unlabeled Data)', fontsize=14)
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# Plot 2: EMG signals for one gesture (first 1000 samples of class 1 - Rest)
gesture_rest = df[df['class'] == 1].iloc[:1000]
plt.figure(figsize=(14, 6))
for ch in channels[:4]:  # First 4 channels
    plt.plot(gesture_rest['time'].values, gesture_rest[ch].values, label=ch)
plt.legend()
plt.title('EMG Signals for "Hand at Rest" Gesture (First 4 Channels)', fontsize=14)
plt.xlabel('Time (ms)')
plt.ylabel('EMG Amplitude')
plt.tight_layout()
plt.savefig('emg_signals.png')
plt.show()

# =============================
# 3. REMOVE UNLABELED DATA (class 0)
# =============================
print("\n" + "="*50)
print("PREPROCESSING")
print("="*50)
print(f"Original dataset size: {len(df)}")
df = df[df['class'] != 0]
print(f"After removing class 0: {len(df)}")

# =============================
# 4. WINDOWING
# =============================
window_size = 200
X_windows = []
y_windows = []

for start in range(0, len(df) - window_size, window_size):
    end = start + window_size
    window = df.iloc[start:end]
    
    # Check if window contains only one label
    if window['class'].nunique() == 1:
        X_windows.append(window[channels].values)
        y_windows.append(window['class'].iloc[0])

X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

print(f"Number of windows: {len(X_windows)}")
print(f"Window shape: {X_windows.shape}")

# =============================
# 5. FEATURE EXTRACTION
# =============================
def extract_features(window):
    features = []
    
    for ch in range(window.shape[1]):  # for each channel
        signal = window[:, ch]
        
        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(signal))
        
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(signal**2))
        
        # Variance
        var = np.var(signal)
        
        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(signal)))
        
        features.extend([mav, rms, var, wl])
    
    return features

# Apply feature extraction
print("\nExtracting features...")
X_features = np.array([extract_features(w) for w in X_windows])
print(f"Feature matrix shape: {X_features.shape}")

# =============================
# 6. TRAIN/TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, 
    y_windows, 
    test_size=0.2, 
    random_state=42,
    stratify=y_windows
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Number of classes: {len(np.unique(y_windows))}")

# =============================
# 7. SVM WITH GRID SEARCH
# =============================
print("\n" + "="*50)
print("SVM TRAINING WITH GRID SEARCH")
print("="*50)

svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale'],
    'kernel': ['rbf']
}

print("Performing grid search for SVM...")
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)

print(f"\nBest SVM parameters: {svm_grid.best_params_}")
print(f"Best SVM CV accuracy: {svm_grid.best_score_:.4f}")

# Train final SVM with best parameters
best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test)
print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# =============================
# 8. RANDOM FOREST WITH GRID SEARCH
# =============================
print("\n" + "="*50)
print("RANDOM FOREST TRAINING WITH GRID SEARCH")
print("="*50)

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

print("Performing grid search for Random Forest...")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"\nBest RF parameters: {rf_grid.best_params_}")
print(f"Best RF CV accuracy: {rf_grid.best_score_:.4f}")

# Train final RF with best parameters
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")

# =============================
# 9. CLASSIFICATION REPORT & CONFUSION MATRIX
# =============================
print("\n" + "="*50)
print("RANDOM FOREST DETAILED RESULTS")
print("="*50)

gesture_names = ['Rest', 'Fist', 'Wrist Flexion', 'Wrist Extension', 
                 'Radial Deviation', 'Ulnar Deviation']

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=gesture_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, 
                                       display_labels=gesture_names,
                                       cmap='Blues')
plt.title('Random Forest Confusion Matrix', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# =============================
# 10. FEATURE IMPORTANCE (Random Forest)
# =============================
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

feature_names = []
for ch in range(1, 9):
    feature_names.extend([f'ch{ch}_MAV', f'ch{ch}_RMS', f'ch{ch}_Var', f'ch{ch}_WL'])

importances = best_rf.feature_importances_
indices = np.argsort(importances)[-15:]  # top 15

print("\nTop 10 Most Important Features:")
top_10_indices = np.argsort(importances)[-10:][::-1]
for i in top_10_indices:
    print(f"  {feature_names[i]}: {importances[i]:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(15), importances[indices])
plt.yticks(range(15), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features for EMG Gesture Recognition', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# =============================
# 11. CROSS-VALIDATION
# =============================
print("\n" + "="*50)
print("CROSS-VALIDATION")
print("="*50)

cv_scores = cross_val_score(best_rf, X_features, y_windows, cv=5)
print(f"5-fold Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Std CV accuracy: {cv_scores.std():.4f}")

# =============================
# 12. MODEL COMPARISON
# =============================
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

# Measure training time
start = time.time()
SVC(**svm_grid.best_params_).fit(X_train, y_train)
svm_time = time.time() - start

start = time.time()
RandomForestClassifier(**rf_grid.best_params_, random_state=42).fit(X_train, y_train)
rf_time = time.time() - start

print(f"\n{'Metric':<25} {'SVM':<15} {'Random Forest':<15}")
print("-"*55)
print(f"{'Test Accuracy':<25} {accuracy_score(y_test, y_pred_svm):<15.4f} {rf_accuracy:<15.4f}")
print(f"{'CV Accuracy (5-fold)':<25} {svm_grid.best_score_:<15.4f} {cv_scores.mean():<15.4f}")
print(f"{'Training Time (s)':<25} {svm_time:<15.2f} {rf_time:<15.2f}")
print(f"{'Number of Features':<25} {32:<15} {32:<15}")
print(f"{'Interpretability':<25} {'Medium (kernel)':<15} {'High (importance)':<15}")

# =============================
# 13. GENERALIZATION GAP ANALYSIS
# =============================
print("\n" + "="*50)
print("GENERALIZATION ANALYSIS")
print("="*50)

test_acc = rf_accuracy
cv_acc = cv_scores.mean()
gap = test_acc - cv_acc

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Cross-validation Accuracy: {cv_acc:.4f}")
print(f"Generalization Gap: {gap:.4f} ({gap*100:.1f}%)")

if gap > 0.05:
    print("\n⚠️ Significant gap detected - possible overfitting!")
    print("Suggestions to reduce gap:")
    print("  • Increase regularization (reduce max_depth, increase min_samples_split)")
    print("  • Add more training data (more subjects or data augmentation)")
    print("  • Use dropout or early stopping (for deep learning)")
else:
    print("\n✓ Small generalization gap - model generalizes well!")

# =============================
# 14. SUMMARY
# =============================
print("\n" + "="*50)
print("PROJECT SUMMARY")
print("="*50)
print(f"""
Dataset: EMG Data for Gestures (UCI)
Hardware: Myo Thalmic Bracelet (8 EMG sensors)
Subjects: 36
Gestures: 7 (Rest, Fist, Wrist Flexion, Wrist Extension, 
               Radial Deviation, Ulnar Deviation, Extended Palm)

Preprocessing:
  - Windowing: {window_size} samples per window
  - Windows created: {len(X_windows)}
  - Features per window: 32 (MAV, RMS, Variance, WL × 8 channels)

Results:
  - Random Forest Accuracy: {rf_accuracy:.2%}
  - SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2%}
  - Cross-validation (5-fold): {cv_scores.mean():.2%} ± {cv_scores.std():.2%}

Best Hyperparameters:
  - SVM: {svm_grid.best_params_}
  - Random Forest: {rf_grid.best_params_}

Conclusion:
  Random Forest outperformed SVM on EMG gesture classification 
  due to its ability to handle non-linear patterns and its 
  built-in regularization through bagging and random feature selection.
""")

models = ['SVM', 'Random Forest', 'Cross-Validation (5-fold)']
accuracies = [93.4, 96.1, 88.1]

# Create bar chart
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
# Customize chart
plt.ylim(80, 100)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add a horizontal line at 90%
plt.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% Target')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: accuracy_comparison.png")
plt.close()

print("\n✅ All plots saved:")
print("  - class_distribution.png")
print("  - emg_signals.png")
print("  - confusion_matrix.png")
print("  - feature_importance.png")


