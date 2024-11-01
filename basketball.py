import streamlit as st
import pandas as pd 
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import requests

st.title("NBA Player Stats Explorer")

st.markdown("""
            This app performs webscraping of NBA player stats data. 
            * **Python libraries :** base64, pandas, streamlit
            * **Data source :** [basketball-reference.com](https://www.basketball-reference.com)
            
            **Don't forget to check the sidebar to interact with the dataset and enhance your user experience.**
            """)

st.sidebar.header('User Input features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(2000,2025))))

#Web scraping of NBA player stats

@st.cache_data

def load_data(year):
    #url = "https://www.basketball-reference.com/leagues/NBA_"+str(year)+"_per_game.html"

    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    html = pd.read_html(response.content, header=0)
    #html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) #Delete repeat ages
    raw.fillna(0)
    playerstats=raw.drop(['Rk'], axis=1) #Rk -- Rank car redondant avec les index de pandas 
    return playerstats

playerstats = load_data(selected_year)

playerstats = playerstats.dropna(subset=['Team'])


#sidebar - Team selection
sorted_unique_team = sorted(playerstats.Team.unique())
selected_teams = st.sidebar.multiselect('Teams', sorted_unique_team, default=sorted_unique_team)

#sidebar - position actuelle
unique_pos = ['C', 'PF', 'SF','PG', 'SG']
selected_pos = st.sidebar.multiselect('POsition', unique_pos, default=unique_pos)


#Filtering data 
df_selected_team = playerstats[(playerstats.Team.isin(selected_teams))&(playerstats.Pos.isin(selected_pos))]
if len(selected_teams)>1 : 
    st.header('Display Player Stats of selected Teams')
else :
    st.header('Display Player Stats of selected Team')

st.write('Data Dimension :' + str(df_selected_team.shape[0])+ ' rows (players) and '+ str(df_selected_team.shape[1]) + ' column')
st.dataframe(df_selected_team) #afficher le dataframe

#Download NBA player stats data
def filedownload(df): 
    csv=df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download = "playerstats.csv"> Download CSV File </a>'
    return href

st.markdown(filedownload(df_selected_team),unsafe_allow_html=True)

##################################################

#HeatMap

# HeatMap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    # Select only numeric columns for correlation
    df_numeric = df.select_dtypes(include=[float, int])
    
    corr = df_numeric.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, mask=mask, vmax=1, square=True, ax=ax)
    st.pyplot(f)



#################################################

import numpy as np
# Initialize the app
st.header("NBA Player Stats Comparison")

# Load the data
data = playerstats

# List of statistics by category
stats_options = {
    "Offensive Performance": ['PTS', 'FG%', '3P%', '2P%', 'FT%'],
    "Defensive Performance": ['STL', 'BLK', 'DRB', 'PF'],
    "Overall Efficiency": ['eFG%', 'TS%', 'PTS'],  # "PTS" and "PER" as indicators of offensive efficiency
    "Playmaking and Versatility": ['AST', 'TOV', 'AST%']

}

# Sidebar - Select statistics category
st.sidebar.header("Select a Statistics Category")
category = st.sidebar.selectbox("Statistics Category", list(stats_options.keys()))
stats = stats_options[category]

# Sidebar - Select players
st.sidebar.header("Select Players to Compare")
all_players = data['Player'].unique()  # List of all players
if selected_year == 2024 : 
    selected_players = st.sidebar.multiselect("Choose Players", all_players, default=['LeBron James', 'Stephen Curry', 'Victor Wembanyama'])
else :
    selected_players = st.sidebar.multiselect("Choose Players", all_players, default=['LeBron James', 'Stephen Curry'])

# Check if at least one player is selected
if len(selected_players) >= 1:
    # Filter data for selected players
    selected_data = data[data['Player'].isin(selected_players)]

    # Normalize data for each statistic (scaling between 0 and 1)
    normalized_data = data.copy()
    for stat in stats:
        min_val = data[stat].min()
        max_val = data[stat].max()
        normalized_data[stat] = (data[stat] - min_val) / (max_val - min_val)

    # Retrieve normalized data for selected players
    selected_normalized_data = normalized_data[normalized_data['Player'].isin(selected_players)]

    # Prepare data for radar chart
    values = []
    for player in selected_players:
        player_stats = selected_normalized_data[selected_normalized_data['Player'] == player][stats].values.flatten().tolist()
        values.append(player_stats)

    # Add the first point again at the end to close the radar chart
    values = [player_stats + [player_stats[0]] for player_stats in values]
    angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False).tolist()
    angles += angles[:1]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Plot lines for each selected player
    for i, player_values in enumerate(values):
        ax.plot(angles, player_values, label=selected_players[i])
        ax.fill(angles, player_values, alpha=0.25)

    # Aesthetic settings
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stats)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)
else:
    st.warning("Please select at least one player to display the radar chart.")

