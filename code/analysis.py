#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 13:03:01 2025

@author: konearounaromeo
"""

# analysis.py - ANALYSE EXPLORATOIRE DES DONNÉES
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import unicodedata
import os 



# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")




# Détection automatique du chemin
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '..', 'data', 'processed-data.csv')

print(f"📂 Chargement depuis : {data_path}")
df = pd.read_csv(data_path)
print(f"Fichier chargé avec {df.shape[0]} lignes et {df.shape[1]} colonnes")
df.dropna()

# ============================================================================
# CALCUL DE LA VARIABLE SUCCESS AU DEBUT
# ============================================================================
# Objectif : Créer la variable de succès avant toute analyse
df['success'] = (df['montant_actuel'] >= df['montant_initial']).astype(int)
print("Variable 'success' creee avec succes")

# ============================================================================
# 2.1. STATISTIQUES DESCRIPTIVES
# ============================================================================
# Objectif : Obtenir un résumé complet des variables quantitatives pour 
# détecter les distributions, outliers et patterns généraux

# Afficher toutes les colonnes et lignes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Afficher toutes les largeurs de texte (évite la troncature des chaînes)
pd.set_option('display.max_colwidth', None)

# Réexécuter l’affichage
print(df.describe(include='all'))




print("="*80)
print("STATISTIQUES DESCRIPTIVES - VARIABLES NUMERIQUES")
print("="*80)

numeric_cols = ['montant_initial', 'montant_actuel', 'nombre_contributions', 
                'nombre_commentaires', 'nombre_videos', 'nombre_images', 
                'nombre_publications']

stats_df = df[numeric_cols].describe(percentiles=[.25, .5, .75, .95, .99])
print(stats_df)

# Calcul du skewness et kurtosis
print("\n" + "="*80)
print("ASYMETRIE (SKEWNESS) ET APLATISSEMENT (KURTOSIS)")
print("="*80)
for col in numeric_cols:
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f"{col:25s} | Skewness: {skew:7.2f} | Kurtosis: {kurt:7.2f}")
    
    
#L’ensemble des variables clés sont très asymétriques et leptokurtiques, ce qui révèle :

#une forte concentration du succès dans un petit nombre de projets,

#une grande diversité des profils de campagnes,

#et la nécessité, pour les analyses économétriques futures, d’envisager des transformations (log, winsorisation) afin de réduire l’influence des valeurs extrêmes..


# ============================================================================
# 2.1.1 TRAITEMENT DES VALEURS EXTRÊMES ET RÉDUCTION DE L’ASYMÉTRIE
# ============================================================================
# Objectif : limiter l'influence des valeurs extrêmes (outliers) et réduire 
# l'asymétrie avant les analyses économétriques futures.



# 1️ Winsorisation à 1% (borne inférieure et supérieure)
for col in numeric_cols:
    df[f'{col}_w'] = winsorize(df[col], limits=[0.01, 0.01])

print("\nWinsorisation effectuée sur 1% des valeurs extrêmes (haut/bas).")

# 2️ Transformation logarithmique pour stabiliser la variance
for col in numeric_cols:
    df[f'{col}_log'] = np.log1p(df[f'{col}_w'])  # log1p(x) = log(1 + x)

print("Transformation logarithmique (log1p) appliquée avec succès.")

# 3️ Vérification rapide de la distribution après traitement
print("\n" + "="*80)
print("STATISTIQUES APRÈS TRANSFORMATION (LOG + WINSORISATION)")
print("="*80)
stats_log = df[[f'{c}_log' for c in numeric_cols]].describe(percentiles=[.25, .5, .75, .95, .99])
print(stats_log)

print("\n" + "="*80)
print("ASYMÉTRIE ET APLATISSEMENT APRÈS TRANSFORMATION")
print("="*80)
for col in numeric_cols:
    skew = df[f'{col}_log'].skew()
    kurt = df[f'{col}_log'].kurtosis()
    print(f"{col+'_log':25s} | Skewness: {skew:7.2f} | Kurtosis: {kurt:7.2f}")



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
    
# Visualisation avant/après transformation
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(numeric_cols[:6]):
    ax = axes.flatten()[i]
    sns.kdeplot(df[col], label='Avant', ax=ax)
    sns.kdeplot(df[f'{col}_log'], label='Après', ax=ax)
    ax.set_title(f"Distribution : {col}")
    ax.legend()
plt.tight_layout()
safe_savefig('winsorisation.png')
plt.show()





# ============================================================================
# 2.1.2 VISUALISATION DES DISTRIBUTIONS GENERALES
# ============================================================================
# Objectif : Visualiser les distributions des variables clés
# Ce qu'on veut vérifier : forme de distribution, log-normalité, outliers

print("\nGeneration des graphiques de distribution...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distributions des Variables Principales', fontsize=16, fontweight='bold')

# 1. Montant initial (échelle log)
axes[0,0].hist(df['montant_initial'], bins=50, edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Montant Initial (€)')
axes[0,0].set_ylabel('Frequence')
axes[0,0].set_title('Distribution du Montant Initial')
axes[0,0].set_xscale('log')

# 2. Montant actuel (échelle log)
axes[0,1].hist(df['montant_actuel'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0,1].set_xlabel('Montant Actuel (€)')
axes[0,1].set_ylabel('Frequence')
axes[0,1].set_title('Distribution du Montant Collecte')
axes[0,1].set_xscale('log')

# 3. Nombre de contributions
axes[1,0].hist(df['nombre_contributions'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1,0].set_xlabel('Nombre de Contributions')
axes[1,0].set_ylabel('Frequence')
axes[1,0].set_title('Distribution des Contributions') 

# 4. Boxplot comparatif
data_box = [df['montant_initial_log'], df['montant_actuel_log']]
axes[1,1].boxplot(data_box, labels=['Montant Initial', 'Montant Actuel'])
axes[1,1].set_ylabel('Montant (€)')
axes[1,1].set_title('Comparaison Montants (Boxplot)')
axes[1,1].set_yscale('log')

plt.tight_layout()
safe_savefig('distributions_generales.png')
plt.show()

print("Graphique sauvegarde : Documents/projet_ulule/demo/images/distributions_generales.png")

# Note : Les histogrammes montrent les distributions originales, 
# tandis que le boxplot compare les montants après log + winsorisation.





# ============================================================================
# 2.2. ANALYSE PAR CATEGORIE
# ============================================================================
# Objectif : Comparer les 4 catégories pour identifier des différences
# structurelles dans les montants, le succès et l'engagement

print("\n" + "="*80)
print("ANALYSE PAR CATEGORIE")
print("="*80)

categories = df['categorie'].unique()


# Statistiques par catégorie
for cat in categories:
    df_cat = df[df['categorie'] == cat]
    print(f"\n{cat.upper()}")
    print("-" * 60)
    print(f"Nombre de projets : {len(df_cat)}")
    print(f"Montant initial moyen : {df_cat['montant_initial'].mean():.2f} €")
    print(f"Montant actuel moyen : {df_cat['montant_actuel'].mean():.2f} €")
    print(f"Mediane contributions : {df_cat['nombre_contributions'].median():.0f}")
    print(f"Proportion de succes : {df_cat['success'].mean()*100:.1f}%")

# Visualisation comparative
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparaison des 4 Categories', fontsize=16, fontweight='bold')

# 1. Boxplot montant initial par catégorie
df.boxplot(column='montant_initial_log', by='categorie', ax=axes[0,0],whis=3)
axes[0,0].set_title('Montant Initial par Categorie')
axes[0,0].set_xlabel('Categorie')
axes[0,0].set_ylabel('Montant Initial (€)')
axes[0,0].set_yscale('log')
plt.sca(axes[0,0])
plt.xticks(rotation=45, ha='right')

# Justification : Même après log-transformation et winsorisation, certains projets restent extrêmes 
# par rapport aux autres, ce qui se reflète dans les boxplots. Ces valeurs sont réalistes et représentent 
# des campagnes exceptionnelles, il est donc pertinent de les conserver pour l'analyse.



# 2. Violin plot montant actuel
sns.violinplot(data=df, x='categorie', y='montant_actuel', ax=axes[0,1])
axes[0,1].set_title('Distribution Montant Collecte par Categorie')
axes[0,1].set_xlabel('Categorie')
axes[0,1].set_ylabel('Montant Actuel (€)')
axes[0,1].set_yscale('log')
plt.sca(axes[0,1])
plt.xticks(rotation=45, ha='right')

# 3. Barplot nombre de projets
cat_counts = df['categorie'].value_counts()
axes[1,0].bar(cat_counts.index, cat_counts.values, edgecolor='black')
axes[1,0].set_title('Repartition des Projets par Categorie')
axes[1,0].set_xlabel('Categorie')
axes[1,0].set_ylabel('Nombre de Projets')
plt.sca(axes[1,0])
plt.xticks(rotation=45, ha='right')

# 4. Proportion de succès par catégorie
success_rate = df.groupby('categorie')['success'].mean() * 100
axes[1,1].bar(success_rate.index, success_rate.values, color='seagreen', edgecolor='black')
axes[1,1].set_title('Proportion de Succes par Categorie (%)')
axes[1,1].set_xlabel('Categorie')
axes[1,1].set_ylabel('Proportion de Succes (%)')
axes[1,1].set_ylim(0, 100)
plt.sca(axes[1,1])
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
safe_savefig('comparaison_categories.png')
plt.show()

print("Graphique sauvegarde : Documents/projet_ulule/demo/images/comparaison_categories.png")

# Interprétation synthétique : Les projets Artisanat & Cuisine et Mode & Design sont nombreux et très souvent réussis, 
# avec un engagement élevé (médiane contributions ~60). Les projets Technologiques, bien que plus ambitieux financièrement, 
# présentent un taux de succès plus faible et moins de contributions, tandis que Santé & Bien-être se situe en profil intermédiaire.





# ============================================================================
# WORDCLOUDS PAR CATEGORIE
# ============================================================================
# Objectif : Générer des nuages de mots pour chaque catégorie
# Ce qu'on veut vérifier : mots-clés distinctifs par catégorie

print("\n" + "="*80)
print("GENERATION DES WORDCLOUDS PAR CATEGORIE")
print("="*80)

# Stopwords français complets
stopwords_fr = set([
    # Articles et déterminants
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'au', 'aux', 'ce', 'cet', 'cette', 'ces',
    'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',

    # Pronoms
    'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'me', 'moi', 'te', 'toi', 'se', 'lui', 'leur', 'soi', 'y', 'en',

    # Verbes auxiliaires et formes courantes
     'suis', 'es', 'est', 'sommes', 'êtes', 'sont',"c'est",
    'avoir', 'ai', 'as', 'a', 'avons', 'avez', 'ont', 'avait', 'avaient', 'auront', 'aurait', 'serait', 'seront', 'étaient',
    'faire', 'fait', 'fais', 'font', 'faisons', 'faisait', 'faites',

    # Prépositions
    'à', 'de', 'dans', 'en', 'par', 'pour', 'vers', 'chez', 'sur', 'sous', 'après', 'avec','avant', 'pendant', 'depuis', 'entre', 'sans', 'selon', 'contre',

    # Conjonctions et mots de liaison
    'et', 'ou', 'donc', 'car', 'mais', 'or', 'ni', 'que', 'quand', 'comme', 'si', 'lorsque', 'puis', 'alors', 'ainsi',

    # Adverbes courants
    'pas', 'plus', 'moins', 'très', 'trop', 'mal', 'aussi', 'encore', 'toujours', 'jamais', 'déjà', 
    'ici', 'là', 'partout', 'ailleurs', 'maintenant', 'hier', 'aujourd’hui', 'demain',

    # Autres mots fonctionnels
    'tout', 'toute', 'tous', 'toutes', 'chaque', 'aucun', 'aucune', 'quelque', 'quelques', 
    'certains', 'certaines', 'autre', 'autres', 'même', 'seul', 'seule', 'cela', 'celui', 'ceux', 'celles', 
    'afin', 'ainsi', 'alors', 'donc', 'comme', 'depuis', 'etc', 'être', 'avoir', 'faire', 'voir', 'dire', 
    'peut', 'peu', 'grand', 'petit', 'autant', 'fois', 'chaque', 'tant', 'plusieurs', 'moins', 'sous', 'trop', 
    'néanmoins', 'notamment', 'tandis', 'lors', 'qui','qu’il', 'qu’elle', 'qu’ils', 'qu’elles', 'qu’on', 'd’un', 'd’une'
])

all_stopwords = STOPWORDS.union(stopwords_fr)

categories = df['categorie'].unique()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Nuages de Mots par Categorie (Titres de Projets)', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, cat in enumerate(categories):
    # Combiner tous les titres de la catégorie
    text = ' '.join(df[df['categorie'] == cat]['titre'].astype(str).values)
    
    # Générer le wordcloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         stopwords=all_stopwords,
                         colormap='viridis',
                         max_words=100).generate(text)
    
    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].set_title(cat, fontsize=14, fontweight='bold')
    axes[idx].axis('off')
    
    print(f"Wordcloud genere pour : {cat}")

plt.tight_layout()
safe_savefig('wordclouds_categories.png',)
plt.show()

print("Graphique sauvegarde : Documents/projet_ulule/demo/images/wordclouds_categories.png")




# Interprétation :
# Les wordclouds révèlent les thèmes dominants dans les titres des projets.
# - Artisanat & cuisine : mots liés à l’expérience locale et à la convivialité (ex : atelier, café, restaurant).
# - Santé & bien-être : vocabulaire centré sur le naturel, le soin et la cosmétique.
# - Mode & design : mise en avant de la création, de l’identité française et du style.
# - Technologie : accent sur l’innovation et les solutions numériques (projet, application, 3D).
# Cela illustre des stratégies narratives différentes selon les catégories, influençant potentiellement le succès des campagnes.


# ============================================================================
# 2.3. ANALYSE GEOGRAPHIQUE
# ============================================================================
# Objectif : Identifier les clusters urbains et leur performance
# Ce qu'on veut vérifier : corrélation entre localisation et succès

print("\n" + "="*80)
print("ANALYSE GEOGRAPHIQUE")
print("="*80)

# Filtrer les villes valides (exclure 'none')
df_geo = df[df['adresse'] != 'none'].copy()

# Top 15 villes
top_villes = df_geo['adresse'].value_counts().head(15)
print("\nTop 15 des villes avec le plus de projets :")
print(top_villes)

# Calculer la proportion de succès par ville (top 15)
ville_stats = df_geo.groupby('adresse').agg({
    'success': ['count', 'mean'],
    'montant_actuel': 'median',
    'montant_initial': 'median'
}).round(2)

ville_stats.columns = ['nb_projets', 'proportion_succes', 'montant_median_actuel', 'montant_median_initial']
ville_stats = ville_stats[ville_stats['nb_projets'] >= 10].sort_values('nb_projets', ascending=False).head(15)

print("\nStatistiques par ville (minimum 10 projets) :")
print(ville_stats)

# Visualisation
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Analyse Geographique des Projets', fontsize=16, fontweight='bold')

# 1. Barplot top villes
axes[0].barh(top_villes.index, top_villes.values, edgecolor='black')
axes[0].set_xlabel('Nombre de Projets')
axes[0].set_title('Top 15 des Villes par Nombre de Projets')
axes[0].invert_yaxis()

# 2. Scatter : nombre projets vs proportion succès
axes[1].scatter(ville_stats['nb_projets'], ville_stats['proportion_succes']*100, 
               s=ville_stats['montant_median_actuel']/10, alpha=0.6, edgecolor='black')
axes[1].set_xlabel('Nombre de Projets')
axes[1].set_ylabel('Proportion de Succes (%)')
axes[1].set_title('Relation entre Volume et Performance (taille = montant median collecte)')
axes[1].grid(alpha=0.3)

# Ajouter les noms des villes
for idx, row in ville_stats.iterrows():
    axes[1].annotate(idx, (row['nb_projets'], row['proportion_succes']*100), 
                    fontsize=8, alpha=0.7)

plt.tight_layout()
safe_savefig('analyse_geographique.png')
plt.show()

print("Graphique sauvegarde : Documents/projet_ulule/demo/images/analyse_geographique.png")

# Interprétation : Les projets sont fortement concentrés dans les grandes villes (notamment Paris et Lyon), 
# où les taux de succès dépassent souvent 95 %. Cela traduit un effet d’agglomération et un accès plus facile 
# aux réseaux de financement participatif. Les montants médians varient selon la taille et la dynamique économique 
# des territoires.




# ============================================================================
# 2.4. ANALYSE TEXTE
# ============================================================================
# Objectif : Analyse des caractéristiques textuelles
# Ce qu'on veut vérifier : impact de la longueur du titre sur le succès

print("\n" + "="*80)
print("ANALYSE DES CARACTERISTIQUES TEXTUELLES")
print("="*80)

# Créer la variable title_len
df['title_len'] = df['titre'].astype(str).str.len()

# Statistiques
print(f"Longueur moyenne des titres : {df['title_len'].mean():.1f} caracteres")
print(f"Longueur mediane : {df['title_len'].median():.0f} caracteres")

# Comparer longueur selon succès
success_titles = df[df['success'] == 1]['title_len']
failed_titles = df[df['success'] == 0]['title_len']

print(f"\nTitres projets reussis : {success_titles.mean():.1f} caracteres (moyenne)")
print(f"Titres projets echoues : {failed_titles.mean():.1f} caracteres (moyenne)")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Analyse de la Longueur des Titres', fontsize=16, fontweight='bold')

# 1. Distribution longueur titres
axes[0].hist([success_titles, failed_titles], bins=30, label=['Succes', 'Echec'], 
            alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Longueur du Titre (caracteres)')
axes[0].set_ylabel('Frequence')
axes[0].set_title('Distribution selon le Succes')
axes[0].legend()

# 2. Boxplot comparatif
axes[1].boxplot([success_titles, failed_titles], labels=['Succes', 'Echec'])
axes[1].set_ylabel('Longueur du Titre (caracteres)')
axes[1].set_title('Comparaison Mediane')

plt.tight_layout()
safe_savefig('analyse_texte.png',)
plt.show()

print("Graphique sauvegarde : Documents/projet_ulule/demo/images/analyse_texte.png")

# Interprétation : Les projets réussis ont des titres plus courts et percutants, 
# tandis que les projets échoués utilisent des titres plus longs. 
# Cela suggère qu’une communication synthétique favorise la réussite d’une campagne.



# ============================================================================
# 2.5. CORRELATIONS
# ============================================================================
# Objectif : Matrice de corrélation des variables numériques
# Ce qu'on veut vérifier : quelles variables sont liées au succès

print("\n" + "="*80)
print("MATRICE DE CORRELATION")
print("="*80)

# Sélection des variables numériques
numeric_vars = ['montant_initial', 'montant_actuel', 'nombre_contributions',
               'nombre_commentaires', 'nombre_videos', 'nombre_images',
               'nombre_publications', 'title_len', 'success']

corr_matrix = df[numeric_vars].corr()

# Afficher les corrélations avec success
print("\nCorrelations avec la variable SUCCESS :")
print(corr_matrix['success'].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Matrice de Correlation (Pearson)', fontsize=16, fontweight='bold')
plt.tight_layout()
safe_savefig('matrice_correlation.png')
plt.show()

print("Graphique sauvegarde : Documents/projet_ulule/demo/images/images/matrice_correlation.png")


# Interprétation :
# Les corrélations montrent une forte liaison entre les contributions et les commentaires,
# traduisant un effet d'engagement communautaire. 
# Les variables de communication (publications, images, vidéos) sont modérément corrélées entre elles
# et positivement liées au succès, suggérant que la visibilité renforce la probabilité de réussite.
# À l'inverse, le montant initial et la longueur du titre sont négativement corrélés avec le succès,
# indiquant qu'une communication claire et des objectifs réalistes favorisent les performances des projets.



# ============================================================================
# PAIR PLOT - ANALYSE MULTIVARIÉE
# ============================================================================
# Objectif : Visualiser les relations entre les variables clés
# Ce qu'on veut vérifier : corrélations, distributions, patterns par catégorie

print("\n" + "="*80)
print("PAIR PLOT - ANALYSE MULTIVARIÉE DES VARIABLES CLÉS")
print("="*80)

# Sélection des variables numériques importantes pour le pair plot
print("\nCréation d'une version simplifiée du pair plot...")

variables_simplifiees = ['montant_initial', 'montant_actuel', 'nombre_contributions', 'success']

plt.figure(figsize=(12, 10))
pair_plot_simple = sns.pairplot(df[variables_simplifiees + ['categorie']], 
                               hue='categorie',
                               diag_kind='kde',
                               palette='Set2',
                               plot_kws={'alpha': 0.7, 's': 40},
                               diag_kws={'alpha': 0.8, 'linewidth': 1.5})

pair_plot_simple.fig.suptitle('PAIR PLOT Simplifié - Variables Principales\nDistributions et Corrélations', 
                             y=1.02, fontsize=16, fontweight='bold')

plt.tight_layout()
safe_savefig('pair_plot_analysis.png')
plt.show()


print("Pair plot simplifié sauvegardé : Documents/projet_ulule/demo/images/pair_plot_analysis.png")
# Interprétation du Pair Plot :
# - Les distributions des montants (initial et actuel) sont très asymétriques, avec quelques valeurs extrêmes.
# - Corrélation positive entre montant_initial et montant_actuel : plus l’objectif est élevé, plus la collecte tend à l’être.
# - Le nombre_contributions est positivement lié au montant_actuel → plus de contributions = plus de fonds levés.
# - La variable success (binaire) est surtout associée aux projets ayant un montant_actuel élevé et un grand nombre de contributions.
# - Certaines catégories (ex : technologie, mode & design) présentent des projets plus performants en moyenne.





# ============================================================================
# 2.6. PRINCIPAL COMPONENT ANALYSIS (PCA) - Cercle de Corrélation
# ============================================================================
# Objectif : Explorer les interactions entre variables et leur contribution
# au succès des projets, pour expliquer les modèles et identifier les patterns




def circle_of_correlations(pc_infos, ebouli, figsize=(8,8)):
    """
    Affiche le cercle des corrélations pour les deux premières composantes principales
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Cercle unité
    circle = plt.Circle((0,0),1, color='g', fill=False, lw=1)
    ax.add_artist(circle)

    # Tracer les flèches pour chaque variable
    for i in range(pc_infos.shape[0]):
        ax.arrow(0, 0, 
                 pc_infos.iloc[i,0], 
                 pc_infos.iloc[i,1], 
                 head_width=0.03, head_length=0.03, color='b', length_includes_head=True)
        ax.text(pc_infos.iloc[i,0]*1.1, pc_infos.iloc[i,1]*1.1, 
                pc_infos.index[i], color='r', fontsize=10, ha='center', va='center')

    # Axes
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

    ax.set_xlabel(f'PC1 ({ebouli[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({ebouli[1]*100:.1f}%)')
    ax.set_title('Cercle de Corrélation - PCA')
    ax.grid(True)

    # Forcer ratio égal + limites symétriques pour X et Y
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    safe_savefig('pca.png')
    plt.show()

    print("Pair plot simplifié sauvegardé : Documents/projet_ulule/demo/images/pca.png")

    


def perform_pca(df, features, n_components=2):
    """
    PCA sur le dataset standardisé
    """
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Composantes et contributions
    pc_infos = pd.DataFrame(pca.components_.T, 
                            columns=[f'PC{i+1}' for i in range(n_components)], 
                            index=features)
    ebouli = pd.Series(pca.explained_variance_ratio_)

    print("\nExplained variance ratio (Ebouli) :")
    print(ebouli)

    # Cercle de corrélation
    circle_of_correlations(pc_infos, ebouli)

    return pc_infos, ebouli, X_pca


# Variables numériques à inclure
pca_features = ['montant_initial', 'nombre_contributions', 'nombre_commentaires',
                'nombre_videos', 'nombre_images', 'nombre_publications', 'title_len']

pc_infos, ebouli, X_pca = perform_pca(df, pca_features)



# Interprétation :
# Le cercle de corrélation montre deux axes principaux :
# - PC1 (~31%) : axe de l’engagement collectif (nombre_contributions, nombre_commentaires).
# - PC2 (~18%) : axe de la communication visuelle (images, vidéos, publications).
# Les variables nombre_contributions et nombre_commentaires sont presque confondues,
# traduisant une forte corrélation : plus un projet est commenté, plus il reçoit de contributions.
# Les efforts visuels et le niveau d’ambition (montant_initial) jouent un rôle secondaire
# mais différenciateur selon la stratégie du porteur de projet.


# ============================================================================
# 2.7. TESTS STATISTIQUES
# ============================================================================
# Objectif : Tests d'hypothèses statistiques

print("\n" + "="*80)
print("TESTS STATISTIQUES")
print("="*80)

# Test 1 : ANOVA - Comparaison montant_actuel entre catégories
print("\n[TEST 1] ANOVA : Montant collecte selon la categorie")
print("-" * 60)
print("H0 : Les moyennes des montants collectes sont identiques entre categories")
print("H1 : Au moins une categorie differe")

categories = df['categorie'].unique()
groups = [df[df['categorie'] == cat]['montant_actuel'].values for cat in categories]
f_stat, p_value = stats.f_oneway(*groups)

print(f"F-statistique : {f_stat:.3f}")
print(f"p-value : {p_value:.4f}")
if p_value < 0.05:
    print("Conclusion : Difference significative detectee (p < 0.05)")
else:
    print("Conclusion : Pas de difference significative (p >= 0.05)")

# Test 2 : Chi2 - Indépendance catégorie et succès
print("\n[TEST 2] Test du Chi2 : Independance Categorie et Succes")
print("-" * 60)
print("H0 : La categorie et le succes sont independants")
print("H1 : Il existe une relation entre categorie et succes")

contingency_table = pd.crosstab(df['categorie'], df['success'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi2 : {chi2:.3f}")
print(f"p-value : {p_value:.4f}")
print(f"Degres de liberte : {dof}")
if p_value < 0.05:
    print("Conclusion : Association significative (p < 0.05)")
else:
    print("Conclusion : Pas d'association significative (p >= 0.05)")

# Interprétation :
# L’ANOVA indique que les montants collectés diffèrent significativement selon les catégories,
# révélant des dynamiques économiques propres à chaque secteur.
# Le test du Chi² montre une dépendance entre la catégorie et le succès,
# confirmant que la typologie du projet influence directement sa probabilité de réussite.
# Ces différences justifient l’intégration de la variable "categorie" dans les futurs modèles explicatifs.




# ============================================================================
# ANALYSE COMPARATIVE DONS vs PREVENTES(avec standardisation)
# ============================================================================
# Objectif : Analyse comparative entre dons avec objectif et preventes
# Ce qu'on veut vérifier : Différences structurelles entre les deux modèles


# Les "préventes" correspondent à des ventes anticipées de produits/services.
# Le montant affiché pour ces projets n'indique pas une collecte financière,
# mais simplement le nombre d'unités à vendre (quantités). Le prix n'est pas connu à l'avance.
# Par conséquent, les métriques financières (montant collecté, % financé, etc.) ne sont
# pas directement comparables avec les "dons avec objectif".
# Seules les métriques structurelles ou en unités (nombre de projets, contributions,
# proportion de succès) sont pertinentes pour la comparaison.


print("\n" + "="*80)
print("ANALYSE COMPARATIVE : DONS AVEC OBJECTIF vs PREVENTES")
print("="*80)

# Séparation selon le type de modèle
dons_objectif = df[(df['type_projet'] == 'euros') & (df['montant_initial'].notna())].copy()
preventes = df[df['type_projet'] == 'prevente'].copy()


print(f"\nREPARTITION :")
print(f"   Dons avec objectif : {len(dons_objectif):,} projets ({len(dons_objectif)/len(df)*100:.1f}%)")
print(f"   Préventes          : {len(preventes):,} projets ({len(preventes)/len(df)*100:.1f}%)")


# Calcul des % financés uniquement pour les dons (non pertinent pour les préventes)
dons_objectif['pct_funded'] = (dons_objectif['montant_actuel'] / dons_objectif['montant_initial']) * 100
preventes['pct_funded'] = np.nan  # Eviter de calculer un pourcentage de "financement" pour les ventes


# Standardisation des variables structurelles

cols_compare = ['nombre_contributions', 'nombre_commentaires', 'nombre_images']

scaler = StandardScaler()
dons_objectif_std = dons_objectif.copy()
preventes_std = preventes.copy()
dons_objectif_std[cols_compare] = scaler.fit_transform(dons_objectif[cols_compare])
preventes_std[cols_compare] = scaler.fit_transform(preventes[cols_compare])


# Statistiques comparatives

print(f"\nSTATISTIQUES COMPARATIVES :")
print(f"\n{'Metrique':<30} {'Dons':>15} {'Préventes':>15} {'Difference':>15}")
print("-" * 78)

metrics = {
    'Proportion de succes (%)': ('success', 'mean', 100),
    'Contributions mediane': ('nombre_contributions', 'median', 1),
    'Contributions moyenne': ('nombre_contributions', 'mean', 1),
    'Commentaires median': ('nombre_commentaires', 'median', 1),
    'Images mediane': ('nombre_images', 'median', 1),
}

for label, (col, func, mult) in metrics.items():
    val_dons = getattr(dons_objectif[col], func)() * mult
    val_prev = getattr(preventes[col], func)() * mult
    diff = val_prev - val_dons
    print(f"{label:<30} {val_dons:>15,.1f} {val_prev:>15,.1f} {diff:>+15,.1f}")


# Visualisations comparatives

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Analyse Comparative : Dons avec Objectif vs Préventes', fontsize=16, fontweight='bold')

# 1. Répartition (pie chart)
sizes = [len(dons_objectif), len(preventes)]
labels = ['Dons avec objectif', 'Préventes']
colors = ['#1f77b4', '#ff7f0e']
axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
axes[0].set_title('Répartition des projets', fontweight='bold')

# 2. Proportion de succès
success_dons = dons_objectif['success'].mean() * 100
success_prev = preventes['success'].mean() * 100
axes[1].bar(labels, [success_dons, success_prev], color=colors, edgecolor='black', alpha=0.8)
axes[1].set_ylabel('Proportion de succès (%)')
axes[1].set_ylim(0, 100)
axes[1].set_title('Proportion de succès', fontweight='bold')
for i, v in enumerate([success_dons, success_prev]):
    axes[1].text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')

# 3. Comparaison des variables structurelles standardisées
# Utilisation de contributions, commentaires et images standardisées
data_y = [dons_objectif_std[cols_compare].mean(axis=1), preventes_std[cols_compare].mean(axis=1)]
axes[2].boxplot(data_y, labels=['Dons', 'Préventes'], patch_artist=True)
axes[2].set_title('Engagement moyen (standardisé)', fontweight='bold')
axes[2].set_ylabel('Valeur standardisée')
colors_bp = ['#1f77b4', '#ff7f0e']
for patch, color in zip(axes[2].artists, colors_bp):
    patch.set_facecolor(color)

plt.tight_layout()
safe_savefig('dons_vs_prevents_standardized.png')
plt.show()
print("Graphique sauvegardé : Documents/projet_ulule/demo/images/dons_vs_prevents_standardized.png")



# Interprétation :
# Dons (~74 %) : succès élevé (~87.6 %), contributions et interactions modérées.
# Préventes (~26 %) : succès quasi total (~99.4 %), contributions et interactions plus fortes.
# Conclusion : modèles distincts, préventes à analyser comme quantité, pas comme financement.




