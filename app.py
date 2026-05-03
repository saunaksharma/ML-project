import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="wide")

TEAMS = ['CSK', 'MI', 'RCB', 'KKR', 'SRH', 'DC', 'PBKS', 'RR']

VENUES = {
    'CSK':  'Chepauk Stadium, Chennai',
    'MI':   'Wankhede Stadium, Mumbai',
    'RCB':  'Chinnaswamy Stadium, Bangalore',
    'KKR':  'Eden Gardens, Kolkata',
    'SRH':  'Rajiv Gandhi Stadium, Hyderabad',
    'DC':   'Feroz Shah Kotla, Delhi',
    'PBKS': 'PCA Stadium, Mohali',
    'RR':   'Sawai Mansingh Stadium, Jaipur',
}

TEAM_COLORS = {
    'CSK': '#F5A623', 'MI': '#004BA0', 'RCB': '#EC1C24',
    'KKR': '#3A225D', 'SRH': '#FF822A', 'DC':  '#0078BC',
    'PBKS': '#ED1B24', 'RR': '#EA1A85',
}

# ─── Data Generation ──────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=600):
    np.random.seed(42)
    rows = []
    for _ in range(n):
        t1, t2 = np.random.choice(TEAMS, 2, replace=False)
        venue = VENUES[t1]
        toss_winner = np.random.choice([t1, t2])
        toss_decision = np.random.choice(['bat', 'field'])

        # Slight realistic biases
        t1_prob = 0.50
        if toss_winner == t1 and toss_decision == 'bat':
            t1_prob += 0.04
        if toss_winner == t2 and toss_decision == 'field':
            t1_prob -= 0.04
        # Home advantage
        t1_prob += 0.05

        winner = t1 if np.random.random() < t1_prob else t2
        rows.append([t1, t2, venue, toss_winner, toss_decision, winner])

    return pd.DataFrame(rows, columns=['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner'])


# ─── Model Training ───────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    le = {
        'team':     LabelEncoder().fit(TEAMS),
        'venue':    LabelEncoder().fit(list(VENUES.values())),
        'decision': LabelEncoder().fit(['bat', 'field']),
        'winner':   LabelEncoder().fit(TEAMS),
    }

    enc = df.copy()
    enc['team1']        = le['team'].transform(df['team1'])
    enc['team2']        = le['team'].transform(df['team2'])
    enc['venue']        = le['venue'].transform(df['venue'])
    enc['toss_winner']  = le['team'].transform(df['toss_winner'])
    enc['toss_decision']= le['decision'].transform(df['toss_decision'])
    enc['winner']       = le['winner'].transform(df['winner'])

    X = enc[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']]
    y = enc['winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, le, acc, X_train, X_test, y_train, y_test


# ─── Load everything ──────────────────────────────────────────────────────────
df = generate_data()
model, le, accuracy, X_train, X_test, y_train, y_test = train_model(df)


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🏏 IPL Match Win Predictor")
st.markdown("**Machine Learning Project — 6th Sem MSIT | Algorithm: Random Forest Classifier**")
st.markdown("---")

# Sidebar — model info
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
    st.caption(f"Training samples : {len(X_train)}")
    st.caption(f"Testing  samples : {len(X_test)}")
    st.caption("Algorithm : Random Forest")
    st.caption("Features  : 5")
    st.markdown("---")
    st.markdown("**Features used:**")
    st.markdown("- Team 1\n- Team 2\n- Venue\n- Toss Winner\n- Toss Decision")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Visualizations", "📋 Dataset"])

# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Match Details")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("🟡 Team 1 (Batting First)", TEAMS, index=0)
    with col2:
        team2_opts = [t for t in TEAMS if t != team1]
        team2 = st.selectbox("🔵 Team 2 (Bowling First)", team2_opts, index=0)

    col3, col4, col5 = st.columns(3)
    with col3:
        venue = st.selectbox("🏟️ Venue", list(VENUES.values()))
    with col4:
        toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
    with col5:
        toss_decision = st.radio("🏏 Toss Decision", ['bat', 'field'], horizontal=True)

    st.markdown("")
    predict_btn = st.button("🔮 Predict Winner", use_container_width=True, type="primary")

    if predict_btn:
        inp = pd.DataFrame({
            'team1':        [le['team'].transform([team1])[0]],
            'team2':        [le['team'].transform([team2])[0]],
            'venue':        [le['venue'].transform([venue])[0]],
            'toss_winner':  [le['team'].transform([toss_winner])[0]],
            'toss_decision':[le['decision'].transform([toss_decision])[0]],
        })

        pred   = model.predict(inp)[0]
        winner = le['winner'].inverse_transform([pred])[0]
        proba  = model.predict_proba(inp)[0]

        classes = list(le['winner'].classes_)
        t1_prob = proba[classes.index(team1)] * 100 if team1 in classes else 0
        t2_prob = proba[classes.index(team2)] * 100 if team2 in classes else 0

        st.markdown("---")
        st.success(f"### 🏆 Predicted Winner: **{winner}**")

        rc1, rc2 = st.columns(2)
        rc1.metric(f"{team1}", f"{t1_prob:.1f}%", delta="Win Probability")
        rc2.metric(f"{team2}", f"{t2_prob:.1f}%", delta="Win Probability")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(5, 2))
        bars = ax.barh([team2, team1], [t2_prob, t1_prob],
                       color=[TEAM_COLORS.get(team2, '#888'), TEAM_COLORS.get(team1, '#888')])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Win Probability (%)")
        ax.set_title("Head-to-Head Win Probability")
        for bar, val in zip(bars, [t2_prob, t1_prob]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ── Tab 2: Visualizations ────────────────────────────────────────────────────
with tab2:
    st.subheader("📈 Data Visualizations")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Win Count by Team**")
        win_counts = df['winner'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = [TEAM_COLORS.get(t, '#999') for t in win_counts.index]
        ax1.bar(win_counts.index, win_counts.values, color=colors, edgecolor='white')
        ax1.set_xlabel("Team")
        ax1.set_ylabel("Wins")
        ax1.set_title("Total Wins per Team")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.markdown("**Toss Decision Distribution**")
        toss_counts = df['toss_decision'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie(toss_counts.values, labels=toss_counts.index, autopct='%1.1f%%',
                colors=['#F5A623', '#004BA0'], startangle=90)
        ax2.set_title("Bat vs Field after Toss")
        st.pyplot(fig2)
        plt.close()

    st.markdown("**Feature Importance**")
    feat_names = ['Team 1', 'Team 2', 'Venue', 'Toss Winner', 'Toss Decision']
    importances = model.feature_importances_
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    bars3 = ax3.barh(feat_names, importances * 100, color='#3A7BD5')
    ax3.set_xlabel("Importance (%)")
    ax3.set_title("Feature Importance in Random Forest Model")
    for bar, val in zip(bars3, importances * 100):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center')
    st.pyplot(fig3)
    plt.close()

    st.markdown("**Toss Winner vs Match Winner**")
    df['toss_helped'] = df['toss_winner'] == df['winner']
    helped_counts = df['toss_helped'].value_counts()
    fig4, ax4 = plt.subplots(figsize=(4, 3))
    ax4.bar(['Toss Winner Won', 'Toss Winner Lost'],
            [helped_counts.get(True, 0), helped_counts.get(False, 0)],
            color=['#2ecc71', '#e74c3c'])
    ax4.set_ylabel("Number of Matches")
    ax4.set_title("Does Winning Toss Help?")
    st.pyplot(fig4)
    plt.close()

# ── Tab 3: Dataset ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📋 Dataset Overview")
    st.markdown(f"Total Records: **{len(df)}** | Features: **5** | Target: `winner`")

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total Matches", len(df))
    col_s2.metric("Teams", len(TEAMS))
    col_s3.metric("Venues", len(VENUES))

    st.dataframe(df.head(30), use_container_width=True)

    st.markdown("**Column Descriptions:**")
    desc = pd.DataFrame({
        'Column': ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner'],
        'Description': [
            'Team batting first',
            'Team bowling first',
            'Stadium where match is played',
            'Team that won the toss',
            'Decision after winning toss (bat/field)',
            'Match winner (Target variable)'
        ],
        'Type': ['Categorical'] * 6
    })
    st.dataframe(desc, use_container_width=True, hide_index=True)
