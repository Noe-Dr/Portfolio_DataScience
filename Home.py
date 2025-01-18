import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Analyse Lexicale CGI",
    page_icon="📚",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("Navigation")
    option = st.selectbox(
        "Choisissez une section",
        ["Accueil", "Analyse Temporelle", "Impact Scientifique", "Visualisation", "Stratégie"]
    )
    
    st.divider()
    
    # Filtres
    st.subheader("Filtres")
    periode = st.slider("Période d'analyse", 2019, 2024, (2019, 2024))
    axes = st.multiselect(
        "Axes de recherche",
        ["SCALE", "DiSCS", "TRACE", "HOSPI"]
    )
    type_doc = st.selectbox(
        "Type de document",
        ["Tous", "Articles", "Thèses", "Rapports"]
    )

# Contenu principal
if option == "Accueil":
    st.title("📚 Analyse Lexicale des Publications CGI")
    st.markdown("""
    ### Présentation
    Cette application permet d'analyser l'évolution des recherches du CGI d'IMT Mines Albi
    sur la période 2019-2024 et de projeter les tendances pour 2025-2030.
    
    ### Fonctionnalités principales
    - Analyse temporelle des publications
    - Étude d'impact scientifique
    - Visualisation des tendances
    - Analyse stratégique
    """)
    
    # Métriques exemple
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Publications", "150", "+12%")
    with col2:
        st.metric("Mots-clés uniques", "324", "-8%")
    with col3:
        st.metric("Chercheurs", "45", "+5%")
        
    # Zone de téléchargement
    st.divider()
    st.subheader("Import des données")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
elif option == "Analyse Temporelle":
    st.title("📈 Analyse Temporelle")
    st.markdown("""
    Cette section permettra de visualiser l'évolution des thématiques dans le temps.
    """)
    
    # Placeholder pour les graphiques
    st.subheader("Évolution des mots-clés")
    st.empty()  # Emplacement pour un futur graphique
    
elif option == "Impact Scientifique":
    st.title("🎯 Impact Scientifique")
    st.markdown("""
    Analyse de l'impact des publications et des collaborations.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Citations par axe")
        st.empty()  # Emplacement pour un futur graphique
    
    with col2:
        st.subheader("Réseau de collaboration")
        st.empty()  # Emplacement pour un futur graphique
        
elif option == "Visualisation":
    st.title("🔍 Visualisation des Données")
    
    viz_type = st.radio(
        "Type de visualisation",
        ["Nuage de mots", "Graphe de relations", "Carte thématique"]
    )
    
    st.empty()  # Emplacement pour la visualisation choisie
    
else:  # Stratégie
    st.title("🎯 Analyse Stratégique")
    st.markdown("""
    ### Objectifs 2025-2030
    
    - Identification des tendances émergentes
    - Recommandations pour le développement futur
    - Analyse des opportunités
    """)
    
    # Zone pour les recommandations
    st.text_area("Ajouter une recommandation", height=100)
    st.button("Soumettre")