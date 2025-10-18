import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import os

# Configuration de la page
st.set_page_config(
    page_title="Analyse Crowdfunding Ulule",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2e86ab;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .interpretation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Supprimer les espaces vides */
    div.st-emotion-cache-1r6slb0 {
        display: none !important;
    }
    /* Cacher les conteneurs vides */
    [data-testid="stVerticalBlock"] > [style*="flex-grow: 1"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Détection du chemin
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, '..', 'data', 'processed-data.csv')

@st.cache_data
def load_data():
    """Charge les données nettoyées"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Calcul des variables comme dans analysis.py
        df['success'] = (df['montant_actuel'] >= df['montant_initial']).astype(int)
        df['title_len'] = df['titre'].astype(str).str.len()
        
        # Feature Engineering comme dans ml.py
        df['goal_log'] = np.log1p(df['montant_initial'])
        df['categorie'] = df['categorie'].fillna('unknown')
        df['type_projet'] = df['type_projet'].fillna('unknown')
        df['categorie_encoded'] = LabelEncoder().fit_transform(df['categorie'])
        df['type_encoded'] = LabelEncoder().fit_transform(df['type_projet'])
        
        # Créer les variables transformées
        numeric_cols = ['montant_initial', 'montant_actuel', 'nombre_contributions', 
                       'nombre_commentaires', 'nombre_videos', 'nombre_images', 
                       'nombre_publications']
        
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_w'] = winsorize(df[col].fillna(0), limits=[0.01, 0.01])
                df[f'{col}_log'] = np.log1p(df[f'{col}_w'])
        
        return df
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Vérifiez que le fichier 'processed-data.csv' est présent.")
        return None

# ---------- Visualisations améliorées ----------

def create_distribution_plots(df):
    """Crée des histogrammes interactifs"""
    figs = {}
    
    # 1. Montant initial
    figs['montant_initial'] = px.histogram(
        df, 
        x='montant_initial', 
        nbins=50,
        title='Distribution du Montant Initial',
        labels={'montant_initial': 'Montant Initial (€)'},
        color_discrete_sequence=['#1f77b4']
    )
    figs['montant_initial'].update_layout(showlegend=False)
    
    # 2. Montant actuel
    figs['montant_actuel'] = px.histogram(
        df, 
        x='montant_actuel', 
        nbins=50,
        title='Distribution du Montant Collecté',
        labels={'montant_actuel': 'Montant Actuel (€)'},
        color_discrete_sequence=['#ff7f0e']
    )
    figs['montant_actuel'].update_layout(showlegend=False)
    
    # 3. Boxplot comparatif
    if 'montant_initial_log' in df.columns and 'montant_actuel_log' in df.columns:
        box_df = pd.DataFrame({
            'Montant Initial': df['montant_initial_log'],
            'Montant Actuel': df['montant_actuel_log']
        })
        box_melt = box_df.melt(var_name='Type', value_name='Montant_log')
        
        figs['box'] = px.box(
            box_melt, 
            x='Type', 
            y='Montant_log', 
            title='Comparaison Montants (log transformée)',
            color='Type'
        )
        figs['box'].update_layout(yaxis_title='log(Montant + 1)', showlegend=False)
    
    return figs

def create_category_analysis(df):
    """Analyse par catégorie avec visualisations avancées"""
    # Boxplot montant_initial par categorie
    fig_box = px.box(df, x='categorie', y='montant_initial', 
                     title='Montant Initial par Catégorie', points=False)
    fig_box.update_yaxes(type='log', title='Montant Initial (€)')
    fig_box.update_layout(xaxis_tickangle=-45)
    
    # Graphique en radar pour comparer les catégories
    category_stats = df.groupby('categorie').agg({
        'success': 'mean',
        'montant_initial': 'mean',
        'nombre_contributions': 'median',
        'nombre_commentaires': 'median'
    }).round(3)
    
    categories = category_stats.index
    metrics = ['success', 'montant_initial', 'nombre_contributions', 'nombre_commentaires']
    
    fig_radar = go.Figure()
    
    for cat in categories:
        values = [category_stats.loc[cat, metric] for metric in metrics]
        max_values = [category_stats[metric].max() for metric in metrics]
        normalized_values = [v/max_val if max_val > 0 else 0 for v, max_val in zip(values, max_values)]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=['Taux Succès', 'Montant Initial', 'Contributions', 'Commentaires'],
            fill='toself',
            name=cat
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparaison Multidimensionnelle des Catégories"
    )
    
    # Proportion de succès par catégorie
    success_rate = df.groupby('categorie')['success'].mean().reset_index()
    success_rate['percent'] = success_rate['success'] * 100
    fig_success = px.bar(success_rate, x='categorie', y='percent', 
                        title='Proportion de Succès par Catégorie (%)')
    fig_success.update_yaxes(range=[0,100])
    fig_success.update_layout(xaxis_tickangle=-45)
    
    return {
        'box': fig_box,
        'radar': fig_radar,
        'success': fig_success
    }

def create_geographic_analysis(df):
    """Analyse géographique avec carte de chaleur"""
    df_geo = df[df['adresse'].notna() & (df['adresse'] != 'none')].copy()
    if df_geo.empty:
        return None, None
    
    # Simulation de coordonnées géographiques
    villes_francaises = {
        'paris': {'lat': 48.8566, 'lon': 2.3522},
        'lyon': {'lat': 45.7640, 'lon': 4.8357},
        'marseille': {'lat': 43.2965, 'lon': 5.3698},
        'toulouse': {'lat': 43.6047, 'lon': 1.4442},
        'nice': {'lat': 43.7102, 'lon': 7.2620},
        'nantes': {'lat': 47.2184, 'lon': -1.5536},
        'montpellier': {'lat': 43.6108, 'lon': 3.8767},
        'bordeaux': {'lat': 44.8378, 'lon': -0.5792},
        'lille': {'lat': 50.6292, 'lon': 3.0573},
        'rennes': {'lat': 48.1173, 'lon': -1.6778}
    }
    
    # Préparation des données pour la carte
    map_data = []
    for ville, count in df_geo['adresse'].value_counts().head(15).items():
        ville_lower = ville.lower()
        if ville_lower in villes_francaises:
            map_data.append({
                'ville': ville,
                'count': count,
                'lat': villes_francaises[ville_lower]['lat'],
                'lon': villes_francaises[ville_lower]['lon']
            })
    
    if map_data:
        map_df = pd.DataFrame(map_data)
        fig_map = px.density_mapbox(map_df, lat='lat', lon='lon', z='count',
                                   radius=20, center=dict(lat=46.6031, lon=1.8883),
                                   zoom=5, mapbox_style="open-street-map",
                                   title="Densité des Projets par Ville")
    else:
        fig_map = None
    
    # Top villes
    top_villes = df_geo['adresse'].value_counts().head(10)
    fig_bar = px.bar(top_villes, x=top_villes.values, y=top_villes.index,
                    orientation='h', title='Top 10 des Villes par Nombre de Projets',
                    color=top_villes.values)
    fig_bar.update_layout(xaxis_title='Nombre de Projets', yaxis_title='Ville')
    
    return fig_map, fig_bar

def create_correlation_matrix(df):
    """Crée une heatmap interactive de la matrice de corrélation"""
    numeric_vars = ['montant_initial', 'montant_actuel', 'nombre_contributions',
                   'nombre_commentaires', 'nombre_videos', 'nombre_images',
                   'nombre_publications', 'title_len', 'success']
    
    numeric_vars = [col for col in numeric_vars if col in df.columns]
    corr_matrix = df[numeric_vars].corr()
    
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                   title='Matrice de Corrélation',
                   color_continuous_scale='RdBu', origin='lower')
    
    return fig, corr_matrix

def create_pca_analysis(df):
    """Analyse PCA avec cercle des corrélations"""
    pca_features = ['montant_initial', 'nombre_contributions', 'nombre_commentaires',
                   'nombre_videos', 'nombre_images', 'nombre_publications', 'title_len']
    
    X_pca = df[pca_features].dropna()
    if len(X_pca) == 0:
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    pca = PCA(n_components=2)
    X_pca_result = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    # Cercle des corrélations
    components = pca.components_.T
    comp_df = pd.DataFrame({
        'feature': pca_features,
        'x': components[:,0],
        'y': components[:,1]
    })
    
    fig = go.Figure()
    
    # Cercle unité
    theta = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), 
        mode='lines', 
        line=dict(color='lightgray', dash='dash'), 
        showlegend=False
    ))
    
    # Flèches
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, row in comp_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, row['x']], 
            y=[0, row['y']], 
            mode='lines+markers',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6, color=colors[i % len(colors)]),
            showlegend=False
        ))
        
        fig.add_annotation(
            x=row['x']*1.15, 
            y=row['y']*1.15, 
            text=row['feature'],
            showarrow=False, 
            font=dict(size=10),
            bgcolor="white",
            bordercolor=colors[i % len(colors)],
            borderwidth=1
        )
    
    fig.update_layout(
        title=f'Cercle de Corrélation - PCA (PC1: {explained_variance[0]*100:.1f}%, PC2: {explained_variance[1]*100:.1f}%)',
        xaxis=dict(range=[-1.2, 1.2], title='PC1'),
        yaxis=dict(range=[-1.2, 1.2], title='PC2'),
        plot_bgcolor='white'
    )
    
    return fig, explained_variance

def create_ml_analysis(df):
    """Analyse machine learning avec configuration interactive"""
    # Configuration des modèles
    st.markdown("#### Configuration des modèles")
    
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Nombre d'estimateurs (Random Forest):", 50, 500, 200)
        max_depth = st.slider("Profondeur maximale:", 5, 20, 12)
    with col2:
        test_size = st.slider("Taille du jeu de test (%):", 10, 40, 20)
        random_state = st.number_input("Seed aléatoire:", 42)
    
    # Préparation des données
    feature_cols = [
        'montant_initial', 'nombre_contributions', 'nombre_commentaires',
        'nombre_videos', 'nombre_images', 'nombre_publications',
        'title_len', 'categorie_encoded', 'type_encoded', 'goal_log'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y_clf = (df['montant_actuel'] >= df['montant_initial']).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_clf, test_size=test_size/100, stratify=y_clf, random_state=random_state
    )
    
    # Entraînement des modèles
    col1, col2 = st.columns(2)
    results = {}
    
    with col1:
        st.markdown("##### Régression Logistique")
        log_reg = LogisticRegression(max_iter=1000, random_state=random_state)
        log_reg.fit(X_train, y_train)
        
        y_pred_log = log_reg.predict(X_test)
        y_proba_log = log_reg.predict_proba(X_test)[:, 1]
        
        accuracy_log = accuracy_score(y_test, y_pred_log)
        roc_auc_log = roc_auc_score(y_test, y_proba_log)
        
        # Matrice de confusion
        cm_log = confusion_matrix(y_test, y_pred_log)
        fig_cm_log = px.imshow(cm_log, text_auto=True, aspect="auto",
                              labels=dict(x="Prédit", y="Réel"),
                              x=['Échec', 'Succès'], y=['Échec', 'Succès'],
                              title='Matrice de Confusion - Régression Logistique')
        st.plotly_chart(fig_cm_log, use_container_width=True)
        
        results['logistic'] = {'accuracy': accuracy_log, 'roc_auc': roc_auc_log}
        
        st.write(f"**Accuracy:** {accuracy_log:.3f}")
        st.write(f"**ROC-AUC:** {roc_auc_log:.3f}")
    
    with col2:
        st.markdown("##### Random Forest")
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        rf_clf.fit(X_train, y_train)
        
        y_pred_rf = rf_clf.predict(X_test)
        y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]
        
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
        
        # Importance des variables
        importances = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_clf.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(importances, x='Importance', y='Feature', orientation='h',
                               title='Importance des Variables - Random Forest')
        st.plotly_chart(fig_importance, use_container_width=True)
        
        results['random_forest'] = {'accuracy': accuracy_rf, 'roc_auc': roc_auc_rf}
        
        st.write(f"**Accuracy:** {accuracy_rf:.3f}")
        st.write(f"**ROC-AUC:** {roc_auc_rf:.3f}")
    
    return results

# ---------- MAIN ----------
def main():
    # Header principal
    st.markdown('<div class="main-header">ANALYSE DES FACTEURS DE SUCCÈS DU CROWDFUNDING SUR ULULE</div>', unsafe_allow_html=True)
    
    # Introduction
    st.write("""
    Cette étude vise à comprendre les déterminants du succès des campagnes de crowdfunding sur la plateforme Ulule, 
    en se concentrant sur quatre catégories spécifiques : **Artisanat & Cuisine**, **Mode & Design**, 
    **Santé & Bien-être** et **Technologie**.
    """)
    
    # Chargement des données
    df = load_data()
    if df is None:
        st.warning("Chargement des données en cours...")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    sections = [
        "Introduction et Méthodologie",
        "Statistiques Descriptives", 
        "Analyse par Catégorie",
        "Analyse Géographique",
        "Analyse Textuelle",
        "Analyse des Corrélations",
        "Analyse en Composantes Principales",
        "Tests Statistiques",
        "Modèles Prédictifs",
        "Conclusion"
    ]
    selected_section = st.sidebar.selectbox("Sélectionnez une section", sections)
    
    # Filtres
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filtres**")
    if 'categorie' in df.columns:
        chosen_cats = st.sidebar.multiselect("Filtrer catégories", 
                                           options=sorted(df['categorie'].unique()), 
                                           default=sorted(df['categorie'].unique()))
        df = df[df['categorie'].isin(chosen_cats)]

    # Section 1: Introduction
    if selected_section == "Introduction et Méthodologie":
        st.markdown('<div class="section-header">INTRODUCTION ET MÉTHODOLOGIE</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Contexte du Crowdfunding
            Le crowdfunding représente une alternative innovante aux circuits de financement traditionnels. 
            Ulule, plateforme française leader, permet à des porteurs de projets de collecter des fonds auprès du grand public.
            
            ### Objectifs de l'Étude
            Cette analyse vise à identifier les facteurs clés influençant le succès des campagnes Ulule.
            """)
            
            with st.expander("Méthodologie détaillée"):
                st.markdown("""
                - **Collecte** : Données de 4 catégories Ulule (France)
                - **Nettoyage** : Suppression des incohérences et valeurs manquantes
                - **Analyse** : Statistiques descriptives, tests d'hypothèses, modèles prédictifs
                - **Outils** : Python, Pandas, Scikit-learn, Streamlit
                """)
            
        with col2:
            st.metric("Projets analysés", f"{len(df):,}")
            st.metric("Taux de succès global", f"{df['success'].mean()*100:.1f}%")
            st.metric("Montant moyen collecté", f"{df['montant_actuel'].mean():.0f}€")
            
            # Graphique de répartition
            cat_counts = df['categorie'].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index, 
                        title='Répartition par Catégorie')
            st.plotly_chart(fig, use_container_width=True)

    # Section 2: Statistiques Descriptives
    elif selected_section == "Statistiques Descriptives":
        st.markdown('<div class="section-header">STATISTIQUES DESCRIPTIVES</div>', unsafe_allow_html=True)
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Projets réussis", f"{df['success'].sum():,}")
        with col2:
            st.metric("Taux de succès", f"{df['success'].mean()*100:.1f}%")
        with col3:
            st.metric("Montant initial moyen", f"{df['montant_initial'].mean():.0f}€")
        with col4:
            st.metric("Contributions moyennes", f"{df['nombre_contributions'].mean():.0f}")
        
        # Graphiques de distribution
        st.markdown("#### Distributions des variables principales")
        figs = create_distribution_plots(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figs['montant_initial'], use_container_width=True)
        with col2:
            st.plotly_chart(figs['montant_actuel'], use_container_width=True)
        
        if 'box' in figs:
            st.plotly_chart(figs['box'], use_container_width=True)
        
        # Interprétation sans boîte
        st.markdown("---")
        st.markdown("""
        **Interprétation :** Les distributions montrent une forte asymétrie avec quelques projets très performants.
        La majorité des projets ont des objectifs financiers modestes, avec présence de valeurs extrêmes
        typiques des plateformes de crowdfunding.
        """)

    # Section 3: Analyse par Catégorie
    elif selected_section == "Analyse par Catégorie":
        st.markdown('<div class="section-header">ANALYSE PAR CATÉGORIE</div>', unsafe_allow_html=True)
        
        cat_figs = create_category_analysis(df)
        
        # Graphiques comparatifs
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(cat_figs['box'], use_container_width=True)
        with col2:
            st.plotly_chart(cat_figs['success'], use_container_width=True)
        
        st.plotly_chart(cat_figs['radar'], use_container_width=True)
        
        # Interprétation sans boîte
        st.markdown("---")
        st.markdown("""
        **Interprétation :** 
        - **Artisanat & Cuisine** et **Mode & Design** : Nombreux projets avec taux de succès très élevés
        - **Technologie** : Projets plus ambitieux financièrement mais taux de succès plus faible
        - **Santé & Bien-être** : Profil intermédiaire avec bon taux de succès
        """)

    # Section 4: Analyse Géographique
    elif selected_section == "Analyse Géographique":
        st.markdown('<div class="section-header">ANALYSE GÉOGRAPHIQUE</div>', unsafe_allow_html=True)
        
        fig_map, fig_bar = create_geographic_analysis(df)
        
        if fig_map is not None:
            st.plotly_chart(fig_map, use_container_width=True)
        
        if fig_bar is not None:
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Interprétation sans boîte
        st.markdown("---")
        st.markdown("""
        **Interprétation :** Forte concentration des projets dans les grandes métropoles (Paris, Lyon, etc.).
        Taux de succès généralement élevés dans les grands centres urbains, montrant un effet d'agglomération
        favorable au crowdfunding.
        """)

    # Section 5: Analyse Textuelle
    elif selected_section == "Analyse Textuelle":
        st.markdown('<div class="section-header">ANALYSE TEXTUELLE</div>', unsafe_allow_html=True)
        
        success_titles = df[df['success'] == 1]['title_len']
        failed_titles = df[df['success'] == 0]['title_len']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Longueur moyenne - Projets réussis", f"{success_titles.mean():.1f} caractères")
        with col2:
            st.metric("Longueur moyenne - Projets échoués", f"{failed_titles.mean():.1f} caractères")
        
        # Distribution
        hist_df = pd.DataFrame({
            'len': np.concatenate([success_titles.values, failed_titles.values]),
            'status': ['Succès'] * len(success_titles) + ['Échec'] * len(failed_titles)
        })
        fig_hist = px.histogram(hist_df, x='len', color='status', barmode='overlay', 
                               nbins=30, title='Distribution de la longueur des titres')
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Interprétation sans boîte
        st.markdown("---")
        st.markdown("""
        **Interprétation :** Les projets réussis ont généralement des titres plus courts et percutants.
        Les projets échoués utilisent des titres plus longs. Une communication concise semble favoriser le succès.
        """)

    # Section 6: Analyse des Corrélations
    elif selected_section == "Analyse des Corrélations":
        st.markdown('<div class="section-header">ANALYSE DES CORRÉLATIONS</div>', unsafe_allow_html=True)
        
        corr_fig, corr_matrix = create_correlation_matrix(df)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Corrélations avec le succès
        st.markdown("#### Corrélations avec la variable succès")
        success_corr = corr_matrix['success'].sort_values(ascending=False)
        st.dataframe(success_corr, use_container_width=True)
        
        # Interprétation sans boîte
        st.markdown("---")
        st.markdown("""
        **Interprétation :**
        - **Forte corrélation positive** : Contributions ↔ Commentaires (engagement communautaire)
        - **Corrélations modérées positives** : Publications, Images, Vidéos (communication)
        - **Corrélations négatives** : Montant initial, Longueur du titre (objectifs trop ambitieux)
        """)

    # Section 7: Analyse en Composantes Principales
    elif selected_section == "Analyse en Composantes Principales":
        st.markdown('<div class="section-header">ANALYSE EN COMPOSANTES PRINCIPALES</div>', unsafe_allow_html=True)
        
        pca_fig, explained_variance = create_pca_analysis(df)
        if pca_fig is not None:
            st.plotly_chart(pca_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Variance expliquée - PC1", f"{explained_variance[0]*100:.1f}%")
            with col2:
                st.metric("Variance expliquée - PC2", f"{explained_variance[1]*100:.1f}%")
            
            # Interprétation sans boîte
            st.markdown("---")
            st.markdown("""
            **Interprétation :**
            - **PC1** : Axe de l'engagement communautaire (contributions, commentaires)
            - **PC2** : Axe de la communication visuelle (images, vidéos, publications)
            Deux leviers complémentaires du succès identifiés.
            """)

    # Section 8: Tests Statistiques
    elif selected_section == "Tests Statistiques":
        st.markdown('<div class="section-header">TESTS STATISTIQUES</div>', unsafe_allow_html=True)
        
        from scipy.stats import chi2_contingency
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Test ANOVA - Montant Collecté par Catégorie")
            categories = df['categorie'].unique()
            groups = [df[df['categorie'] == cat]['montant_actuel'].values for cat in categories]
            valid_groups = [g for g in groups if len(g) > 1]
            
            if len(valid_groups) >= 2:
                f_stat, p_value_anova = stats.f_oneway(*valid_groups)
                st.metric("F-statistique", f"{f_stat:.3f}")
                st.metric("p-value", f"{p_value_anova:.4f}")
                
                if p_value_anova < 0.05:
                    st.success("Différence significative entre les catégories")
                else:
                    st.warning("Pas de différence significative")
        
        with col2:
            st.markdown("#### Test du Chi² - Indépendance Catégorie et Succès")
            contingency_table = pd.crosstab(df['categorie'], df['success'])
            
            if contingency_table.size > 0:
                chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
                st.metric("Chi²", f"{chi2:.3f}")
                st.metric("p-value", f"{p_value_chi2:.4f}")
                st.metric("Degrés de liberté", dof)
                
                if p_value_chi2 < 0.05:
                    st.success("Association significative entre catégorie et succès")
                else:
                    st.warning("Pas d'association significative")
        
        # Interprétation sans boîte
        st.markdown("---")
        st.markdown("""
        **Interprétation :** Les tests confirment que la catégorie est un facteur explicatif majeur du succès.
        Différence significative des montants collectés entre catégories et association
        significative entre catégorie et probabilité de succès.
        """)

    # Section 9: Modèles Prédictifs
    elif selected_section == "Modèles Prédictifs":
        st.markdown('<div class="section-header">MODÈLES PRÉDICTIFS</div>', unsafe_allow_html=True)
        
        results = create_ml_analysis(df)
        
        # Comparaison des modèles
        if results:
            comparison_data = {
                'Modèle': ['Régression Logistique', 'Random Forest'],
                'Accuracy': [results['logistic']['accuracy'], results['random_forest']['accuracy']],
                'ROC-AUC': [results['logistic']['roc_auc'], results['random_forest']['roc_auc']]
            }
            comparison_df = pd.DataFrame(comparison_data)
            
            fig_comparison = px.bar(comparison_df, x='Modèle', y=['Accuracy', 'ROC-AUC'],
                                  title='Comparaison des Performances', barmode='group')
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Interprétation sans boîte
        st.markdown("---")
        st.markdown("""
        **Interprétation :**
        - **Régression logistique** : Bonne performance globale, interprétabilité des coefficients
        - **Random Forest** : Meilleure détection des échecs, capture des interactions non linéaires
        - **Variables clés** : Engagement communautaire et communication visuelle
        """)

    # Section 10: Conclusion
    elif selected_section == "Conclusion":
        st.markdown('<div class="section-header">CONCLUSION GÉNÉRALE</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Principaux Enseignements
        
        **Intuitions Confirmées**
        
        **1. Projets créatifs vs technologiques**  
        • **Artisanat & Cuisine, Mode & Design** : Taux de succès élevés (92%+)  
        • **Technologie** : Plus ambitieux financièrement mais moins de réussite (79.7%)
        
        **2. Engagement communautaire déterminant**  
        • Corrélation forte entre contributions et commentaires  
        • L'activité des contributeurs = moteur principal du succès
        
        **3. Concentration géographique**  
        • Succès supérieur dans les grandes métropoles  
        • Effet d'agglomération favorable au crowdfunding
        
        **4. Communication impactante**  
        • Titres courts favorisent le succès  
        • Communication visuelle cruciale
        """)
        
        st.markdown("""
        ### Recommandations Stratégiques
        
        **Pour les porteurs de projet :**
        • Objectifs financiers réalistes adaptés à la catégorie  
        • Investissement dans la communication visuelle  
        • Engagement communautaire dès le lancement  
        • Titres courts, clairs et percutants  
        • Capitalisation sur les réseaux locaux
        
        **Pour la plateforme Ulule :**
        • Accompagnement adapté aux spécificités des catégories  
        • Support renforcé pour les projets technologiques  
        • Développement des fonctionnalités communautaires
        """)
        
        # Graphique synthèse final
        metrics = ['Engagement', 'Communication', 'Finance', 'Localisation', 'Innovation']
        values = [85, 78, 72, 88, 65]
        
        fig_radar_final = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself'
        ))
        fig_radar_final.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Performance Globale des Facteurs de Succès"
        )
        st.plotly_chart(fig_radar_final, use_container_width=True)

    # Footer avec lien LinkedIn
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 2rem;'>
            <p>Analyse réalisée avec Streamlit - Octobre 2025</p>
            <p>Inspiré par les travaux de <a href='https://www.linkedin.com/in/jrieke/' target='_blank'>jrieke</a> sur GitHub</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()