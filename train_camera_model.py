#!/usr/bin/env python3
"""
Train Camera ML Model (Random Forest)
Dataset: drowsiness_data_camera.xlsx
Features: metric_PERCLOS, metric_BlinkRate, blink_duration_mean
Label: drowsiness_level (1=Alert, 2=Drowsy, 3=Very Drowsy)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings
import os

warnings.filterwarnings('ignore')


def main():
    """Main training pipeline."""
    print("\n" + "🚀"*35)
    print("CAMERA ML MODEL TRAINING PIPELINE")
    print("🚀"*35)
    
    # ============================================
    # DATA PATH
    # ============================================
    data_path = r'C:\Users\nikhi\drowsiness_detection_ros2\src\drowsiness_detection_pkg\drowsiness_detection\drowsiness_data_camera.xlsx'
    
    # LOAD DATA
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_excel(data_path)
    print(f"✅ Loaded {len(df)} samples")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # PREPARE DATA
    print("\n" + "="*70)
    print("📊 PREPARING DATA")
    print("="*70)
    
    # Select features and label
    X = df[['metric_PERCLOS', 'metric_BlinkRate', 'blink_duration_mean']].values
    y = df['drowsiness_level'].values
    
    features = ['metric_PERCLOS', 'metric_BlinkRate', 'blink_duration_mean']
    print(f"✅ Features: {features}")
    print(f"✅ Label column: drowsiness_level")
    
    # Analyze labels
    unique_labels = np.unique(y)
    print(f"\n📋 Unique labels: {unique_labels}")
    print(f"   1 = Alert")
    print(f"   2 = Drowsy")
    print(f"   3 = Very Drowsy")
    
    # Convert multi-class to binary (0=Alert, 1=Drowsy+VeryDrowsy)
    print(f"\n🔄 Converting to binary classification...")
    y_binary = np.array([0 if label == 1 else 1 for label in y])
    
    print(f"\nLabel distribution:")
    unique, counts = np.unique(y_binary, return_counts=True)
    for label, count in zip(unique, counts):
        pct = (count / len(y_binary)) * 100
        label_name = "Alert" if label == 0 else "Drowsy"
        print(f"  {label_name}: {count} samples ({pct:.1f}%)")
    
    # TRAIN MODEL
    print("\n" + "="*70)
    print("🤖 TRAINING MODEL")
    print("="*70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print(f"\n✅ Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✅ Features scaled (StandardScaler)")
    
    # Train Random Forest
    print(f"\n🚀 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    print(f"✅ Model trained successfully!")
    
    # Feature importance
    print(f"\n📊 Feature Importance:")
    for feat, importance in zip(features, model.feature_importances_):
        print(f"  {feat:20s}: {importance:.4f}")
    
    # EVALUATE MODEL
    print("\n" + "="*70)
    print("📈 MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Training metrics
    print(f"\n✅ TRAINING SET:")
    print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_train, y_train_pred):.4f}")
    
    # Testing metrics
    print(f"\n✅ TEST SET:")
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    
    # Confusion Matrix
    print(f"\n📊 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"  [[TN  FP]")
    print(f"   [FN  TP]]")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Alert', 'Drowsy']))
    
    # SAVE MODEL
    print("\n" + "="*70)
    print("💾 SAVING MODEL")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/model_camera_rf.pkl'
    scaler_path = 'models/model_camera_rf_scaler.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # Verify
    print(f"\n🔍 Verifying saved files:")
    print(f"  Model size: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"  Scaler size: {os.path.getsize(scaler_path) / 1024:.2f} KB")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Metrics:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Test ROC-AUC:  {test_auc:.4f}")
    print(f"\n✅ Models saved in 'models/' folder")
    print(f"✅ Ready to use with ML Aggregator Node!")


if __name__ == "__main__":
    main()