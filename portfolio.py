import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Portfolio - Personal Projects", layout="centered")

# Titre principal
st.title("Portfolio - Personal Projects")

# Description
st.write(
    """
    Welcome to my interactive portfolio! Here is a selection of my projects. 
    Click on the links to explore them in detail.
    """
)

# Section pour le projet 1
st.header(" Machine Learning Application: Stock Price Analysis")
st.write(
    """
    A machine learning application utilizing **LSTM (Long Short-Term Memory)** models to analyze stock 
    price movements. This project features an interactive user interface for visualizing and predicting 
    stock trends.
    """
)
st.markdown("[View the project ➡️](https://portfoliodatascience-noe-dreau-stockpriceproject.streamlit.app)", unsafe_allow_html=True)

# Section pour le projet 2
st.header("🏀 NBA Data Analysis: Interactive Heatmaps")
st.write(
    """
    An interactive dashboard for analyzing NBA team performance across various game metrics. 
    It also includes player comparisons based on offensive and defensive skills.
    """
)
st.markdown("[View the project ➡️](https://portfoliodatascience-noe-dreau-basketballproject.streamlit.app)", unsafe_allow_html=True)

# Section pour les futurs projets
st.header("Upcoming Projects")
st.write("Stay tuned for more exciting projects to be added here soon!")

# Footer
st.write("---")
st.write("[Contact me](mailto:noe.dreau@mines-albi.fr)")
