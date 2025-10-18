#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 10:25:31 2025

@author: konearounaromeo
"""


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os 

# Détection automatique du chemin
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '..', 'data', 'processed-data.csv')

print(f"Chargement depuis : {data_path}")
df = pd.read_csv(data_path)
print(f"Fichier chargé avec {df.shape[0]} lignes et {df.shape[1]} colonnes")
df.dropna()

# --- Configuration automatique du dossier d'enregistrement des figures ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_PATH, '..', 'demo', 'images')

# Crée le dossier s’il n’existe pas
os.makedirs(IMG_DIR, exist_ok=True)

def safe_savefig(filename, *args, **kwargs):
    """Sauvegarde une figure automatiquement dans demo/images/."""
    full_path = os.path.join(IMG_DIR, filename)
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight', *args, **kwargs)
    print(f" Graphique sauvegardé : {full_path}")


# =============================================================================
# PRÉPARATION DES VARIABLES EXPLICATIVES
# =============================================================================
print("\n" + "="*80)
print("PRÉPARATION DES FEATURES POUR RÉGRESSION ET CLASSIFICATION")
print("="*80)



# ------------------------------
# Feature Engineering
# ------------------------------
# 1️ Longueur du titre → mesure de l'effort rédactionnel et de communication
df['title_len'] = df['titre'].astype(str).str.len()

# 2️ Transformation logarithmique du montant initial → normalise la dispersion
df['goal_log'] = np.log1p(df['montant_initial'])

# 3️ Encodage des catégories et du type de projet
df['categorie_encoded'] = LabelEncoder().fit_transform(df['categorie'])
df['type_encoded'] = LabelEncoder().fit_transform(df['type_projet'])

# ------------------------------
# Sélection des variables
# ------------------------------
feature_cols = [
    'montant_initial',        # objectif financier de départ
    'nombre_contributions',   # nombre de contributeurs (intensité du soutien)
    'nombre_commentaires',    # niveau d’interaction communautaire
    'nombre_videos',          # qualité du contenu visuel
    'nombre_images',          # attractivité marketing
    'nombre_publications',    # suivi du projet dans le temps
    'title_len',              # effort rédactionnel
    'categorie_encoded',      # nature du projet
    'type_encoded',           # type de modèle (don / prévente)
    'goal_log'                # intensité financière ajustée (log)
]

# Dataset pour la régression et classification
X = df[feature_cols]
y_reg = df['montant_actuel']
y_clf = (df['montant_actuel'] >= df['montant_initial']).astype(int)

print(f"Features sélectionnées : {feature_cols}")
print(f"Taille dataset : X={X.shape}, y_reg={y_reg.shape}, y_clf={y_clf.shape}")

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# =============================================================================
# BASELINE MODEL : RÉGRESSION LOGISTIQUE
# =============================================================================
print("\n" + "="*80)
print("RÉGRESSION LOGISTIQUE : MODÈLE EXPLICATIF DU SUCCÈS")
print("="*80)

# Split train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_clf, test_size=0.2, stratify=y_clf, random_state=42
)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
y_proba_log = log_reg.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred_log)
roc_auc = roc_auc_score(y_test, y_proba_log)

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\n" + classification_report(y_test, y_pred_log, digits=3))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion - Régression Logistique')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.tight_layout()
safe_savefig('reglog.png')
plt.show()
print("Graphique sauvegardé : Documents/projet_ulule/demo/images/reglog.png")


# Matrice de confusion interprétation :
# - 106 vrais échecs correctement identifiés (TN)
# - 52 échecs mal classés comme succès (FP)
# - 1524 vrais succès correctement identifiés (TP)
# - 8 succès mal classés comme échec (FN)
# => Le modèle excelle à détecter les succès, mais a plus de difficultés à repérer les échecs,
#    ce qui reflète l'asymétrie du dataset.


# ------------------------------------------------------------
# Interprétation :
# • La régression logistique (Accuracy ≈ 0.9645, ROC-AUC ≈ 0.971) distingue bien succès et échec,
#   mais détecte moins bien les échecs (recall classe 0 ≈ 0.67).
# ------------------------------------------------------------


# =============================================================================
# MODÈLE NON LINÉAIRE COMPARATIF : RANDOM FOREST
# =============================================================================
print("\n" + "="*80)
print("RANDOM FOREST : CLASSIFICATION DU SUCCÈS (MODÈLE NON LINÉAIRE)")
print("="*80)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)

print(f"Accuracy : {accuracy_rf:.4f}")
print(f"ROC-AUC  : {roc_auc_rf:.4f}")

importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='coolwarm')
plt.title('Importance des Variables - Random Forest')
plt.tight_layout()
safe_savefig('rdf.png')
plt.show()
print("Graphique sauvegardé : Documents/projet_ulule/demo/images/rdf.png")




# ------------------------------------------------------------
# Interprétation :
# • La Random Forest (Accuracy ≈ 0.971, ROC-AUC ≈ 0.9825) améliore la détection des échecs
#   et capture les interactions non linéaires entre variables.
# • Les variables les plus importantes : nombre_contributions, nombre_commentaires, nombre_images, goal_log.
# • En résumé : logistique = interprétation, Random Forest = robustesse prédictive.
# ------------------------------------------------------------


