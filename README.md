### Recall-Optimized K-Nearest Neighbors (KNN) for Supervised Tumor Malignancy Classification

A clean, structured supervised machine learning project implementing the K-Nearest Neighbors (KNN) classification algorithm via scikit-learn to identify tissue malignancy. Because missing a malignant diagnosis (False Negative) has high clinical stakes, this pipeline includes a hyperparameter optimization landscape that specifically targets Recall during cross-validation.
## 📌 Performance Metrics Summary

Following optimal neighborhood calibration, the system outputs the following validation metrics when evaluated against the final unseen testing partition:

    Selected Optimal Hyperparameter (Best K): 11 Neighbors

    Validation Optimization Metric: Validation Recall Optimization (Range: K = 1 to 30)

    Test Classification Accuracy: 98.25%

    Test Sensitivity / Recall Score: 96.15% (Only 1 missed malignant case across test set)

    Test Precision Score: 96.15%

    Test Balanced F1 Score: 96.15%

    Training Set Accuracy Baseline: 97.14% (Evaluated over pooled Train + Val matrix)

## 🛠️ Pipeline Architecture & Validation Strategy

The code uses a robust step-by-step approach to safeguard against data leakage and handle distance scaling sensitivities:
# 1. Sequential Multi-Cohort Split Matrix Setup

Rather than a traditional random split, the dataset is parsed into three distinct validation arrays using index splicing boundaries:

    Training Partition: First 60% of data (Indices: 0 -> 341)

    Validation Tuning Partition: Next 20% of data (Indices: 341 -> 455)

    Independent Test Partition: Final 20% of data (Indices: 455 -> 569)

# 2. Isotropic Standard Feature Scaling

KNN utilizes Euclidean multi-dimensional spatial distances to identify neighbors:
d(p,q)=i=1∑n​(pi​−qi​)2​

Because unscaled variable magnitudes can skew geometric distances, StandardScaler is fitted exclusively on the training split and applied downward to prevent data leakage.
# 3. Recall Tuning Optimization Landscape

The system runs an evaluation loop evaluating odd and even parameters (K=1→30) against the validation split. Instead of optimizing for raw accuracy, it optimizes for Recall to minimize false negatives in the clinical predictions:
Recall=True Positives (TP)+False Negatives (FN)True Positives (TP)​

The loop identifies K = 11 as the optimal configuration for capturing malignant profiles.


# 4. Iterative Hyperparameter Recall Optimization Loop
best_k = 1
best_recall = 0

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    rec = recall_score(y_val, knn.predict(X_val))
    print(f"K={k:>2}  |  Validation Recall = {rec*100:.2f}%")
    if rec > best_recall:
        best_recall = rec
        best_k      = k

print(f"--> Selected Optimal Hyperparameter: Best K = {best_k}")

# 5. Matrix Re-Pooling and Evaluation Final Run
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_full, y_full)

p_test = knn_final.predict(X_test)

# 6. Diagnostic Evaluation Logs Print Block
print("=" * 50)
print(f"   FINAL MATRIX PERFORMANCE SUMMARY (OPTIMAL K = {best_k})   ")
print("=" * 50)
print(f"Train Accuracy Baseline : {accuracy_score(y_full, knn_final.predict(X_full))*100:.2f}%")
print(f"Test Accuracy Score    : {accuracy_score(y_test, p_test)*100:.2f}%")
print("-" * 50)
print(f"Validation Precision   : {precision_score(y_test, p_test)*100:.2f}%")
print(f"Validation Recall      : {recall_score(y_test, p_test)*100:.2f}%")
print(f"Validation F1 Score    : {f1_score(y_test, p_test)*100:.2f}%")
print("=" * 50)

# 7. Visualization Plot Matrix
cm = confusion_matrix(y_test, p_test)
disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix — KNN (K={best_k})')
plt.show()

## 📋 Direct Console Output Display

When executed, the script outputs the optimization path and final test metrics directly to your terminal screen:
Plaintext

K= 1  |  Validation Recall = 88.46%
K= 2  |  Validation Recall = 80.77%
...
K=11  |  Validation Recall = 96.15%
...
K=30  |  Validation Recall = 88.46%
--> Selected Optimal Hyperparameter: Best K = 11

==================================================
   FINAL MATRIX PERFORMANCE SUMMARY (OPTIMAL K = 11)   
==================================================
Train Accuracy Baseline : 97.14%
Test Accuracy Score    : 98.25%
--------------------------------------------------
Validation Precision   : 96.15%
Validation Recall      : 96.15%
Validation F1 Score    : 96.15%
==================================================

## 📊 Matrix Error Verification Analysis

The model generates the following classification breakdown over the unseen test partition (N=119):
Plaintext

Confusion Matrix Array:
[[87  1]   --> [True Benign,  False Malignant]
 [ 1 25]]  --> [False Benign, True Malignant]

Interpretation & Clinical Context

    True Benign (87) / True Malignant (25): The model correctly classified 112 out of 119 patient samples.

    False Positive (1): Only 1 patient with a benign mass was flagged with a malignant signature, which safely routes them to standard follow-up testing.

    False Negative (1): Crucially, the model's recall optimization loop successfully limited missed malignant cases to just 1 out of 26 total cancer profiles. This balance underscores the practical value of tuning hyperparameters to align with clinical risk management goals.
