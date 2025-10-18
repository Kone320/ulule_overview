#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 12:42:57 2025

@author: konearounaromeo
"""


"""
ANALYSE DES DONNÉES - PRÉTRAITEMENT DES DONNÉES ULULE (FRANCE)
Objectif : Nettoyer les données issues du scraping pour ne conserver
que les projets cohérents, complets et appartenant aux 4 catégories
d'intérêt : Santé & bien-être, Artisanat & cuisine, Mode & design, Technologie.
"""

# Importation des bibliothèques 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from scipy import stats
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import unicodedata
import os


# Détection automatique du chemin
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '..', 'data', 'ulule_4_categories_france.csv')

print(f"Chargement depuis : {data_path}")
df = pd.read_csv(data_path)
print(f"ichier chargé avec {df.shape[0]} lignes et {df.shape[1]} colonnes")




print(f"la dimmension de la base de données est:{df.shape}")
print(f"\nVARIABLES:\n{list(df.columns)}")

# Types de données
print(f"\nTypes de données:\n{df.dtypes}")

# Aperçu des données
df.head()

# Valeurs manquantes
print(f"\nvaleurs manquantes par colonnes:\n{df.isna().sum()}")
df.shape

# Suppression des lignes sans titre de projet
# Un projet sans titre ne peut pas être identifié ni analysé.
# Il est donc logique de supprimer ces observations.
df = df.dropna(subset=["titre"])
df = df[df["titre"].str.strip() != ""]
df.shape

# On remplace les villes manquantes par none car par défini par les auteurs de la collecte 
df['adresse'] = df['adresse'].replace('', 'none').fillna('none')
df['adresse'].isna().sum()
df.shape

# Affichage des catégories uniques pour vérification
valeurs_uniques_liste = df['categorie'].unique().tolist()
print(valeurs_uniques_liste)

# ============================================================================
# NETTOYAGE COMPLET DES CHAINES DE CARACTÈRES
# ============================================================================
#  Corriger l'encodage, uniformiser la casse et nettoyer les caractères spéciaux

def clean_text(s):
    """
    Nettoie complètement une chaîne de caractères :
    - Correction de l'encodage
    - Conversion en minuscules
    - Suppression des accents
    - Nettoyage des espaces et caractères spéciaux
    """
    if not isinstance(s, str):
        return s
    
    # Correction de l'encodage
    try:
        s = s.encode('latin1').decode('utf-8')
    except:
        pass
    
    # Conversion en minuscules
    s = s.lower()
    
    # Suppression des accents et caractères spéciaux
    s = unicodedata.normalize('NFKD', s).encode('ascii', errors='ignore').decode('utf-8')
    
    # Nettoyage des espaces multiples et caractères spéciaux en début/fin
    s = ' '.join(s.split())  # Supprime les espaces multiples
    s = s.strip()  # Supprime les espaces en début/fin
    
    return s
#certes perdus les ponctuation et accent... mais on avance

# Application du nettoyage aux colonnes texte
print("Nettoyage des colonnes texte...")
df['adresse'] = df['adresse'].apply(clean_text)
df['titre'] = df['titre'].apply(clean_text)
df['categorie'] = df['categorie'].apply(clean_text)

# Remplacement des valeurs manquantes
df['adresse'] = df['adresse'].fillna('none')

print("Nettoyage des colonnes texte terminé")
print("Exemples d'adresses après nettoyage :", df[df['adresse'] != 'none']['adresse'].unique()[:10])


# Vérification du résultat
print(df['categorie'].unique().tolist())


# Filtrage des catégories d'intérêt
# On ne conserve que les projets appartenant aux quatre domaines ciblés.
# Les autres catégories, vides ou non conformes, sont supprimées.
categories_cibles = ["sante & bien-etre", "artisanat & cuisine", "mode & design", "technologie"]

# Filtrage des lignes appartenant à ces catégories
df = df[df['categorie'].isin(categories_cibles)]

# Vérification du résultat
print("Catégories conservées :", df['categorie'].unique().tolist())
print("Nombre de lignes restantes :", len(df))

# Valeurs manquantes
print(f"\nvaleurs manquantes par colonnes:\n{df.isna().sum()}")
df.shape

# Nettoyage des montants (initial et actuel)
# - montant_initial vide → dons sans objectif → données inutilisables économiquement
# - montant_actuel vide → projet non lancé ou erreur de scraping
# - montant_actuel = 0 avec contributions > 0 → incohérence → suppression
df = df.dropna(subset=["montant_initial", "montant_actuel"])

# Conversion des montants en numérique pour cohérence
df["montant_initial"] = pd.to_numeric(df["montant_initial"], errors="coerce")
df["montant_actuel"] = pd.to_numeric(df["montant_actuel"], errors="coerce")

# Suppression des lignes incohérentes (montant_actuel = 0 mais contributions > 0)
df["nombre_contributions"] = pd.to_numeric(df["nombre_contributions"], errors="coerce")
df = df[~((df["montant_actuel"] == 0) & (df["nombre_contributions"] > 0))]

# Valeurs manquantes
print(f"\nvaleurs manquantes par colonnes:\n{df.isna().sum()}")
df.shape

# Nettoyage des contributions et commentaires
# - cellules vides : projet sans interaction
# - >5000 commentaires : probable erreur de scraping
df = df.dropna(subset=["nombre_contributions", "nombre_commentaires"])
df = df[df["nombre_commentaires"] <= 5000]
df.shape

# Séparation en 2 catégories :
# 1. "dons_objectif" : dons avec objectif financier défini
# 2. "preventes" : préventes avec quantités vendues
dons_objectif = df[(df['type_projet'] == 'euros') & (df['montant_initial'].notna())]
preventes = df[df['type_projet'] == 'prevente']

# Création d'une variable pour distinguer la logique du projet
df['logique_modele'] = df['type_projet'].map({
    'euros': 'financement_don',
    'prevente': 'vente_produit'
})


print(f"Dons avec objectif : {len(dons_objectif)}")
print(f"Préventes : {len(preventes)}")

# sauvegarde du fichier propre 

# Crée le bon chemin absolu pour /data/
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)  # crée le dossier s’il n’existe pas

# Sauvegarde le fichier dans data/
data_path = os.path.join(DATA_DIR, 'processed-data.csv')
df.to_csv(data_path, index=False)
print(f"Fichier sauvegardé : {data_path}")



