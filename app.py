import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Animal Conservation AI", layout="wide")

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #ffffff;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.prediction {
    font-size: 32px;
    font-weight: bold;
    color: #ff4b4b;
}
section[data-testid="stSidebar"] {
    background-color: #111318;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("🧭 Navigation")
st.sidebar.info("Adjust inputs and predict extinction risk")

# ----------------------------
# TITLE
# ----------------------------
st.markdown("""
# 🧠 Animal Conservation AI  
### Predict extinction risk using Machine Learning 🌍
""")

# ----------------------------
# ABOUT CARD
# ----------------------------
st.markdown("""
<div class="card">
<h3>🌍 About This AI</h3>
<p>
This model predicts the conservation status of animal species using ecological and biological factors 
such as habitat range, predator count, lifespan, and reproduction patterns.
</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("2last.pkl")
feature_order = model.feature_names_in_

# ----------------------------
# MAPPINGS
# ----------------------------
diet_map = {0:"Herbivore",1:"Carnivore",2:"Omnivore",3:"Insectivore"}
social_map = {0:"Unknown",1:"Solitary",2:"Pair",3:"Small group",4:"Herd / Pack"}

target_map = {
    0:"Least Concern",
    1:"Near Threatened",
    2:"Vulnerable",
    3:"Endangered",
    4:"Critically Endangered",
    5:"Extinct in the Wild",
    6:"Extinct"
}

# ----------------------------
# INPUT CARD
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    height = st.number_input("Height (cm)", 0.0, 500.0, 120.0)
    weight = st.number_input("Weight (kg)", 0.0, 5000.0, 180.0)
    lifespan = st.number_input("Lifespan (years)", 0.0, 100.0, 15.0)

    diet = st.selectbox("Diet", list(diet_map.keys()), format_func=lambda x: diet_map[x])
    avg_speed = st.number_input("Average Speed (km/h)", 0.0, 200.0, 40.0)

    social = st.selectbox("Social Structure", list(social_map.keys()), format_func=lambda x: social_map[x])

with col2:
    gestation = st.number_input("Gestation Period (days)", 0.0, 700.0, 100.0)
    top_speed = st.number_input("Top Speed (km/h)", 0.0, 200.0, 80.0)
    offspring = st.number_input("Offspring per Birth", 0.0, 10.0, 2.0)

    habitat = st.slider("Habitat Count", 0, 10, 3)
    predators = st.slider("Predator Count", 0, 15, 2)
    countries = st.slider("Countries Count", 0, 200, 10)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# PREDICT BUTTON
# ----------------------------
if st.button("🔍 Predict"):

    data = [[
        height, weight, lifespan, diet, avg_speed,
        social, gestation, top_speed, offspring,
        habitat, predators, countries
    ]]

    df = pd.DataFrame(data, columns=feature_order)

    with st.spinner("🧠 AI is analyzing..."):
        pred = model.predict(df)[0]
        probs = model.predict_proba(df)[0]

    # ----------------------------
    # GRAPH DATA
    # ----------------------------
    class_labels = [target_map[i] for i in model.classes_]

    plot_df = pd.DataFrame({
        "Class": class_labels,
        "Probability": probs
    }).sort_values(by="Probability", ascending=True)

    # ----------------------------
    # OUTPUT LAYOUT
    # ----------------------------
    colA, colB = st.columns([1,1])

    # LEFT → RESULT
    with colA:
        st.markdown(f"""
        <div class="card">
        <div class="prediction">🎯 {target_map[pred]}</div>
        <p>AI Prediction Result</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔝 Top Predictions")
        top_indices = probs.argsort()[-3:][::-1]

        for i in top_indices:
            st.write(f"{target_map[model.classes_[i]]} → {probs[i]*100:.2f}%")

        # 🔥 Confidence Bar
        st.markdown("### 📊 Confidence Level")
        st.progress(float(max(probs)))

    # RIGHT → GRAPH
    with colB:
        sns.set_theme(style="darkgrid")

        fig, ax = plt.subplots(figsize=(5, 3))

        colors = ["#888888"] * len(plot_df)
        pred_index = list(plot_df["Class"]).index(target_map[pred])
        colors[pred_index] = "#FF4B4B"

        ax.barh(plot_df["Class"], plot_df["Probability"], color=colors)

        for i, v in enumerate(plot_df["Probability"]):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", va='center', fontsize=8)

        ax.set_title("Prediction Confidence", fontsize=11)
        ax.set_xlabel("")

        st.pyplot(fig, use_container_width=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Built with Machine Learning • Random Forest Model")