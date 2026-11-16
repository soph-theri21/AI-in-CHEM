# -*- coding: utf-8 -*-
"""Project1_AICHEM_SST.ipynb
"""

!pip install rdkit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

data_train = "/content/drive/MyDrive/AI in CHEM/Tox21_SR_ARE_train.csv"
data_test = "/content/drive/MyDrive/AI in CHEM/Tox21_SR_ARE_test.csv"


df_train = pd.read_csv(data_train)
df_test = pd.read_csv(data_test)


print("Training dataset shape:")
print(df_train.shape)

print("\nTest dataset shape:")
print(df_test.shape)

print("\nTraining columns:")
print(df_train.columns.tolist())

print("\nTest columns:")
print(df_test.columns.tolist())


SMILES_COLUMN = "smiles"
TARGET_COLUMN = "are_tox"

print("\nMissing SMILES:")
print(df_train[SMILES_COLUMN].isna().sum())

print("\nMissing SR_ARE labels:")
print(df_train[TARGET_COLUMN].isna().sum())

df_train_clean = df_train.dropna(
    subset=[SMILES_COLUMN, TARGET_COLUMN]
).copy()

print("\nDataset shape after removing missing values:")
print(df_train_clean.shape)

print("\nTarget values:")
print(df_train_clean[TARGET_COLUMN].value_counts(dropna=False))

#Clean Test Data
print("\nMissing test SMILES:")
print(df_test[SMILES_COLUMN].isna().sum())

print("\nMissing test SR_ARE labels:")
print(df_test[TARGET_COLUMN].isna().sum())

df_test_clean = df_test.dropna(
    subset=[SMILES_COLUMN, TARGET_COLUMN]
).copy()

print("\nTest dataset shape after removing missing values:")
print(df_test_clean.shape)

print("\nTest target values:")
print(df_test_clean[TARGET_COLUMN].value_counts(dropna=False))

def smiles_to_mol(smiles):

    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(smiles)

    return mol

def calc_descriptors(smiles):

    mol = smiles_to_mol(smiles)

    if mol is None:
        return {
            "MolWt": None,
            "MolLogP": None,
            "HBD": None,
            "HBA": None,
            "TPSA": None,
            "RingCount": None,
            "NumRotBonds": None,
            "FractionCSP3": None,
            "NumAromaticRings": None,
            "FormalCharge": None
        }

    return {

        # Molecular size
        "MolWt":
            Descriptors.MolWt(mol),

        # Lipophilicity
        "MolLogP":
            Descriptors.MolLogP(mol),

        # Hydrogen bonding
        "HBD":
            Lipinski.NumHDonors(mol),

        "HBA":
            Lipinski.NumHAcceptors(mol),

        # Polarity
        "TPSA":
            rdMolDescriptors.CalcTPSA(mol),

        # Ring structure
        "RingCount":
            Lipinski.RingCount(mol),

        # Molecular flexibility
        "NumRotBonds":
            Lipinski.NumRotatableBonds(mol),

        # Carbon hybridization / molecular shape
        "FractionCSP3":
            rdMolDescriptors.CalcFractionCSP3(mol),

        # Aromatic character
        "NumAromaticRings":
            rdMolDescriptors.CalcNumAromaticRings(mol),

        # Charge
        "FormalCharge":
            Chem.GetFormalCharge(mol)
    }

train_descriptor_series = df_train_clean[SMILES_COLUMN].apply(
    calc_descriptors
)

train_descriptor_df = pd.DataFrame(
    list(train_descriptor_series),
    index=train_descriptor_series.index
)

test_descriptor_series = df_test_clean[SMILES_COLUMN].apply(
    calc_descriptors
)

test_descriptor_df = pd.DataFrame(
    list(test_descriptor_series),
    index=test_descriptor_series.index
)


print("\nTraining descriptors:")
print(train_descriptor_df.head())

print("\nTest descriptors:")
print(test_descriptor_df.head())

# Remove molecules that RDKit won't process

train_valid_indices = train_descriptor_df.dropna().index

train_descriptor_df = train_descriptor_df.loc[
    train_valid_indices
]

df_train_clean = df_train_clean.loc[
    train_valid_indices
]


test_valid_indices = test_descriptor_df.dropna().index

test_descriptor_df = test_descriptor_df.loc[
    test_valid_indices
]

df_test_clean = df_test_clean.loc[
    test_valid_indices
]


print("\nValid training molecules:")
print(len(df_train_clean))

print("\nValid test molecules:")
print(len(df_test_clean))

selected_features = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "NumRotBonds",
    "FractionCSP3",
    "NumAromaticRings",
    "FormalCharge"
]

X_train = train_descriptor_df[selected_features]
y_train = df_train_clean[TARGET_COLUMN]


X_test = test_descriptor_df[selected_features]
y_test = df_test_clean[TARGET_COLUMN]


print("\nTraining feature shape:")
print(X_train.shape)

print("\nTest feature shape:")
print(X_test.shape)

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            random_state=21
        )
    )
])

parameter_grid = {

    # C controls the penalty for classification errors
    "svm__C": [
        0.1,
        1,
        10
    ],

    # Compare linear and radial-basis-function kernels
    "svm__kernel": [
        "linear",
        "rbf"
    ],

    # Gamma mainly matters for the RBF kernel
    "svm__gamma": [
        "scale",
        0.1,
        0.01
    ],

    "svm__class_weight": [
        None,
        "balanced"
    ]

}
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=21
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)


print("\nRunning GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)

print("\nBest hyperparameters:")
print(grid_search.best_params_)

print("\nBest mean cross-validation ROC-AUC:")
print(grid_search.best_score_)


# Save best model
best_model = grid_search.best_estimator_

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

columns_to_display = [
    "params",
    "mean_train_score",
    "mean_test_score",
    "std_test_score"
]

print("\nGridSearchCV results:")
print(
    grid_results[columns_to_display]
    .sort_values(
        "mean_test_score",
        ascending=False
    )
)

train_predictions = best_model.predict(
    X_train
)

test_predictions = best_model.predict(
    X_test
)

train_scores = best_model.decision_function(
    X_train
)

test_scores = best_model.decision_function(
    X_test
)


train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

train_precision = precision_score(
    y_train,
    train_predictions
)

train_recall = recall_score(
    y_train,
    train_predictions
)

train_f1 = f1_score(
    y_train,
    train_predictions
)

train_roc_auc = roc_auc_score(
    y_train,
    train_scores
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions
)

test_recall = recall_score(
    y_test,
    test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions
)

test_roc_auc = roc_auc_score(
    y_test,
    test_scores
)

print("\n==============================")
print("EXTERNAL TEST CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        test_predictions
    )
)

test_confusion_matrix = confusion_matrix(
    y_test,
    test_predictions
)

print("\nExternal Test Confusion Matrix:")
print(test_confusion_matrix)

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_scores
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"SVM ROC curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve for Tox21 SR_ARE Classification"
)

plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n==============================")
print("OVERFITTING COMPARISON")
print("==============================")

cv_roc_auc = grid_search.best_score_


print(f"Training accuracy:         {train_accuracy:.3f}")
print(f"External test accuracy:    {test_accuracy:.3f}")

print()

print(f"Training ROC-AUC:          {train_roc_auc:.3f}")
print(f"Cross-validation ROC-AUC:  {cv_roc_auc:.3f}")
print(f"External test ROC-AUC:     {test_roc_auc:.3f}")

print()

print(
    f"Train - CV AUC difference: "
    f"{train_roc_auc - cv_roc_auc:.3f}"
)

print(
    f"Train - test AUC difference: "
    f"{train_roc_auc - test_roc_auc:.3f}"
)

print("\n==============================")
print("MODEL SUMMARY")
print("==============================")

print("Number of training compounds:", len(X_train))
print("Number of test compounds:", len(X_test))
print("Number of molecular features:", X_train.shape[1])
print("Selected features:")

for feature in selected_features:
    print(feature)

print("\nBest model parameters:")
print(grid_search.best_params_)
print("\nCross-validation ROC-AUC:")
print(grid_search.best_score_)
print("\nFinal test ROC-AUC:")
print(test_roc_auc)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr_train, tpr_train, _ = roc_curve(y_train, train_scores)
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, _ = roc_curve(y_test, test_scores)
roc_auc_test = auc(fpr_test, tpr_test)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr_train, 
    tpr_train, 
    color='blue', 
    lw=2, 
    label=f'Train ROC curve (AUC = {roc_auc_train:.3f})'
)

plt.plot(
    fpr_test, 
    tpr_test, 
    color='darkorange', 
    lw=2, 
    label=f'Test ROC curve (AUC = {roc_auc_test:.3f})'
)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()