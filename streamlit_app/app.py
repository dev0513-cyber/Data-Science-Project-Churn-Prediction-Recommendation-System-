"""
Customer Churn Prediction — Streamlit Dashboard
Full UI for Data Science Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.churn_utils import ChurnFeatureEngineer
import joblib
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DS — Churn & Movie Recommendations System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  .stMetric label { font-size: 14px !important; }
  .risk-high   { background:#ffebee; border-left:5px solid #f44336; padding:12px; border-radius:6px; }
  .risk-medium { background:#fff8e1; border-left:5px solid #ff9800; padding:12px; border-radius:6px; }
  .risk-low    { background:#e8f5e9; border-left:5px solid #4caf50; padding:12px; border-radius:6px; }
  .section-header { font-size:22px; font-weight:700; color:#1976d2; margin-bottom:10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_churn_model():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "models", "churn_pipeline.pkl")
    path = os.path.normpath(path)  # resolves the ".." cleanly

    if not os.path.exists(path):
        st.error(f"❌ Model file not found at: `{path}`")
        st.info(
            "Files in models/ dir: "
            + str(os.listdir(os.path.join(base, "..", "models")))
        )
        return None

    return joblib.load(path)


@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    df = pd.read_csv(os.path.join(base, "telco_churn.csv"))
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)
    return df


@st.cache_data
def load_movies():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    movies = pd.read_csv(os.path.join(base, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(base, "tmdb_5000_credits.csv"))
    credits.rename(columns={"movie_id": "id"}, inplace=True)
    df = movies.merge(credits, on="id", how="left")
    return df


model = load_churn_model()
df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.title("DS Project")
    st.markdown("**All 10 Tasks Implemented**")
    page = st.radio(
        "📁 Navigate to:",
        [
            "🏠 Home / EDA Dashboard",
            "🔮 Churn Predictor",
            "📊 Batch Analysis",
            "🎬 Movie Recommender",
            "📋 Project Overview",
        ],
    )
    st.divider()
    st.markdown("**Tech Stack:**")
    st.markdown("🐍 Python | 🤖 XGBoost | 📦 scikit-learn")
    st.markdown("⚡ FastAPI | 🎈 Streamlit | 🔍 SHAP")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EDA Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home / EDA Dashboard":
    st.title("📊 Telecom Churn — EDA Dashboard")
    st.markdown("**Dataset:** IBM Telco Customer Churn | 7,043 customers | 21 features")

    # KPI Cards
    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    churn_r = churned / total * 100
    avg_rev = df[df["Churn"] == "No"]["MonthlyCharges"].mean()
    lost_rev = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}", f"{churn_r:.1f}%")
    c3.metric("Avg Monthly Revenue (Retained)", f"${avg_rev:.0f}")
    c4.metric("Monthly Revenue at Risk", f"${lost_rev:,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Churn by Contract Type")
        ct = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index()
        )
        ct.columns = ["Contract", "Churn Rate (%)"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#f44336", "#ff9800", "#4caf50"]
        ax.barh(ct["Contract"], ct["Churn Rate (%)"], color=colors)
        ax.set_xlabel("Churn Rate (%)")
        ax.axvline(churn_r, color="gray", linestyle="--", label=f"Avg {churn_r:.1f}%")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Tenure Distribution by Churn")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, color in [("No", "#4caf50"), ("Yes", "#f44336")]:
            ax.hist(
                df[df["Churn"] == label]["tenure"],
                bins=25,
                alpha=0.6,
                label=f"Churn={label}",
                color=color,
                edgecolor="white",
            )
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Monthly Charges vs Churn")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df.boxplot(
            column="MonthlyCharges",
            by="Churn",
            ax=ax,
            boxprops=dict(color="#1976d2"),
            medianprops=dict(color="red"),
        )
        ax.set_title("")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Monthly Charges ($)")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("#### Internet Service Churn Breakdown")
        ct2 = pd.crosstab(df["InternetService"], df["Churn"])
        ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ct2_pct.plot(kind="bar", ax=ax, color=["#4caf50", "#f44336"], edgecolor="white")
        ax.set_ylabel("%")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(["Stay", "Churn"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### 💡 Business Insights")
    i1, i2, i3 = st.columns(3)
    i1.info("📌 Month-to-month customers churn at **42%** vs 11% for annual contracts")
    i2.warning("⚠️ Fiber optic customers churn at **41.9%** — pricing/quality issue")
    i3.success("✅ Customers with 2+ services churn **60% less** than single-service")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Churn Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.title("🔮 Customer Churn Predictor")
    st.markdown(
        "Fill in customer details and get instant churn prediction with recommendations."
    )

    if model is None:
        st.error("❌ Model not found. Please run `06_End_to_End_Pipeline.ipynb` first!")
        st.stop()

    with st.form("predict_form"):
        st.markdown("#### 👤 Customer Demographics")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        senior = c2.selectbox("Senior Citizen", ["0 - No", "1 - Yes"])
        partner = c3.selectbox("Partner", ["Yes", "No"])
        c4, c5, c6 = st.columns(3)
        dependents = c4.selectbox("Dependents", ["No", "Yes"])
        tenure = c5.slider("Tenure (months)", 0, 72, 12)
        contract = c6.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        st.markdown("#### 📱 Services")
        s1, s2, s3 = st.columns(3)
        phone = s1.selectbox("Phone Service", ["Yes", "No"])
        multi = s2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = s3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        s4, s5, s6 = st.columns(3)
        security = s4.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = s5.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device = s6.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        s7, s8, s9 = st.columns(3)
        tech = s7.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = s8.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies_s = s9.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

        st.markdown("#### 💳 Billing")
        b1, b2, b3 = st.columns(3)
        billing = b1.selectbox("Paperless Billing", ["Yes", "No"])
        payment = b2.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = b3.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.5)
        total_c = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, float(monthly * tenure), 10.0
        )

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        import pandas as pd

        customer = {
            "gender": gender,
            "SeniorCitizen": int(senior[0]),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies_s,
            "Contract": contract,
            "PaperlessBilling": billing,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_c,
        }
        inp_df = pd.DataFrame([customer])
        prob = float(model.predict_proba(inp_df)[0][1])
        pred = int(model.predict(inp_df)[0])

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Prediction", "⚠️ WILL CHURN" if pred else "✅ WILL STAY")
        r2.metric("Churn Probability", f"{prob * 100:.1f}%")
        risk = "HIGH 🔴" if prob >= 0.7 else ("MEDIUM 🟡" if prob >= 0.4 else "LOW 🟢")
        r3.metric("Risk Level", risk)

        # Probability gauge
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh(
            ["Churn Risk"],
            [prob],
            color="#f44336"
            if prob >= 0.7
            else ("#ff9800" if prob >= 0.4 else "#4caf50"),
            height=0.4,
        )
        ax.barh(["Churn Risk"], [1 - prob], left=[prob], color="#e0e0e0", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Churn Probability: {prob * 100:.1f}%")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### 💡 Retention Recommendations")

        recs = []
        if contract == "Month-to-month":
            recs.append(
                "📋 Offer discounted <b>annual contract</b> — reduces churn by 70%"
            )
        if tenure <= 12:
            recs.append(
                "🎁 <b>Early lifecycle customer</b> — trigger welcome retention campaign"
            )
        if monthly > 70:
            recs.append("💰 High charges — offer <b>loyalty discount or bundle</b>")
        if internet == "Fiber optic" and security == "No":
            recs.append(
                " 🔒  Upsell <b>Online Security</b> — increases service stickiness"
            )
        if payment == "Electronic check":
            recs.append(
                "🏦 Encourage <b>auto-payment</b> setup — reduces churn risk 15%"
            )

        if not recs:
            recs.append("✅ Customer appears stable — maintain regular engagement")

        # 🔥 Card UI
        for r in recs:
            st.markdown(
                f"""
            <div style="
            background-color: #1e1e1e;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 10px;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            ">
                {r}
            </div>
            """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Batch Analysis":
    st.title("📊 Batch Churn Analysis")
    st.markdown(
        "Upload a CSV or Excel file to predict churn for all customers at once."
    )

    # ---------------- UI ---------------- #
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Upload customer file", type=["csv", "xlsx"])

        # 🔥 Reset sample mode if new file uploaded
        if uploaded:
            st.session_state["use_sample"] = False
            st.session_state["run_pred"] = False

    with col2:
        st.markdown("**Expected columns:**")
        st.code("tenure, MonthlyCharges, TotalCharges,\nContract, InternetService, ...")

        if st.button("Use sample data (test on telco_churn.csv)"):
            st.session_state["use_sample"] = True
            st.session_state["run_pred"] = False

    use_sample = st.session_state.get("use_sample", False)

    # ---------------- LOAD DATA ---------------- #
    data = None

    if use_sample:
        data = df.drop("Churn", axis=1, errors="ignore").head(100)
        actual = df.head(100)["Churn"].values
        st.info("Using sample dataset (first 100 rows)")

    elif uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                data = pd.read_csv(uploaded)
            elif uploaded.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded)
            else:
                st.error("Unsupported file format")
                st.stop()

            st.success(f"Loaded file: {uploaded.name}")

            # Clean column names
            data.columns = data.columns.str.strip()

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        actual = None

    # ---------------- RUN BUTTON ---------------- #
    if data is not None:
        if st.button("🚀 Run Predictions"):
            st.session_state["run_pred"] = True

    run_pred = st.session_state.get("run_pred", False)

    # ---------------- MODEL PREDICTION ---------------- #
    if model is not None and data is not None and run_pred:
        with st.spinner("Running predictions..."):
            data["TotalCharges"] = pd.to_numeric(
                data.get("TotalCharges", 0), errors="coerce"
            ).fillna(data.get("MonthlyCharges", 50))

            data_clean = data.drop(["customerID"], axis=1, errors="ignore")

            probs = model.predict_proba(data_clean)[:, 1]
            preds = model.predict(data_clean)

        # ---------------- RESULTS ---------------- #
        data["Churn_Probability"] = (probs * 100).round(1)
        data["Churn_Prediction"] = ["CHURN" if p else "STAY" for p in preds]
        data["Risk_Level"] = [
            "HIGH" if p >= 0.7 else ("MEDIUM" if p >= 0.4 else "LOW") for p in probs
        ]

        # ---------------- METRICS ---------------- #
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Customers", len(data))

        c2.metric("Predicted Churners", int(preds.sum()), f"{preds.mean() * 100:.1f}%")

        c3.metric(
            "Monthly Revenue at Risk",
            f"${data[preds == 1]['MonthlyCharges'].sum():,.0f}"
            if "MonthlyCharges" in data.columns
            else "N/A",
        )

        # ---------------- DISPLAY TABLE ---------------- #
        display_data = data[
            [
                "Contract",
                "tenure",
                "MonthlyCharges",
                "Churn_Probability",
                "Risk_Level",
                "Churn_Prediction",
            ]
        ].copy()

        display_data = display_data.sort_values(by="Churn_Probability", ascending=False)

        MAX_ROWS = 1000
        show_data = display_data.head(MAX_ROWS)

        if len(display_data) > MAX_ROWS:
            st.info(
                f"Showing top {MAX_ROWS} high-risk customers out of {len(display_data)}"
            )

        styled_df = show_data.style.background_gradient(
            subset=["Churn_Probability"], cmap="RdYlGn_r"
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=350,
        )

        # ---------------- DOWNLOAD ---------------- #
        csv = data.to_csv(index=False)

        st.download_button(
            "⬇️ Download Results CSV", csv, "churn_predictions.csv", "text/csv"
        )
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Movie Recommender
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎬 Movie Recommender":
    st.title("🎬 Hybrid Movie Recommendation System")
    st.markdown("Content-Based + Collaborative Filtering | TMDB 5000 Movies")

    @st.cache_data
    def build_rec_engine():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        movies_df = load_movies()

        def safe_eval(x):
            try:
                return ast.literal_eval(x)
            except:  # noqa: E722
                return []

        def extract(lst, key="name", n=None):
            items = safe_eval(lst) if isinstance(lst, str) else lst
            names = [i[key] for i in items if key in i]
            return names[:n] if n else names

        def get_dir(crew_str):
            for c in safe_eval(crew_str) if isinstance(crew_str, str) else []:
                if c.get("job") == "Director":
                    return c["name"].replace(" ", "")
            return ""

        movies_df["genres_l"] = movies_df["genres"].apply(extract)
        movies_df["cast_l"] = movies_df["cast"].apply(lambda x: extract(x, n=3))
        movies_df["director"] = movies_df["crew"].apply(get_dir)
        movies_df["overview"] = movies_df["overview"].fillna("")
        movies_df["soup"] = movies_df.apply(
            lambda r: (
                f"{' '.join(r['genres_l'])} {' '.join([c.replace(' ', '') for c in r['cast_l']])} {r['director'] * 3} {r['overview']}"
            ),
            axis=1,
        )

        tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
        mat = tfidf.fit_transform(movies_df["soup"])
        cos_s = cosine_similarity(mat, mat)
        idx_map = pd.Series(
            movies_df.index, index=movies_df["title_x"]
        ).drop_duplicates()
        return movies_df, cos_s, idx_map

    with st.spinner("Building recommendation engine... (~20s first time)"):
        mdf, cos_s, idx_map = build_rec_engine()

    movie_list = sorted(mdf["title_x"].dropna().unique().tolist())

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "🎬 Choose a movie you like:",
            movie_list,
            index=movie_list.index("The Dark Knight")
            if "The Dark Knight" in movie_list
            else 0,
        )
    with col2:
        top_n = st.slider("# Recommendations", 5, 20, 10)

    if st.button("🚀 Get Recommendations", use_container_width=True):
        if selected not in idx_map:
            st.error("Movie not found")
        else:
            idx = idx_map[selected]
            sims = list(enumerate(cos_s[idx]))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
            rec_idx = [s[0] for s in sims]
            sims_score = [s[1] for s in sims]

            recs = (
                mdf[["title_x", "genres_l", "vote_average", "popularity", "overview"]]
                .iloc[rec_idx]
                .copy()
            )
            recs["similarity"] = [round(s, 3) for s in sims_score]
            recs.columns = [
                "Title",
                "Genres",
                "TMDB Rating",
                "Popularity",
                "Overview",
                "Similarity",
            ]

            seed_info = mdf[mdf["title_x"] == selected].iloc[0]
            with st.expander(f"📽️ About '{selected}'", expanded=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.markdown(f"**Genres:** {', '.join(seed_info.get('genres_l', []))}")
                c2.metric("TMDB Rating", f"{seed_info.get('vote_average', 0):.1f}/10")
                c3.metric("Popularity", f"{seed_info.get('popularity', 0):.0f}")

            st.markdown(f"### 🎯 Top {top_n} Movies Similar to '{selected}'")
            for i, row in recs.iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 1.5, 1, 1])
                    c1.markdown(f"**{row['Title']}**")
                    c2.markdown(f"*{', '.join(row['Genres'][:3])}*")
                    c3.markdown(f"⭐ {row['TMDB Rating']:.1f}")
                    c4.markdown(f"🎯 {row['Similarity']:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Project Overview
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Project Overview":
    st.title("📋 Data Science Project — All 10 Tasks")
    st.markdown("Complete project with all 5 levels implemented.")

    tasks = [
        ("Level 1", "Task 1", "Data Cleaning & EDA", "01_EDA_Data_Cleaning.ipynb"),
        (
            "Level 1",
            "Task 2",
            "Basic ML Models (LR, RF, XGB)",
            "02_Basic_ML_Models.ipynb",
        ),
        ("Level 2", "Task 3", "Debug & Improve Flawed Code", "03_Debug_Improve.ipynb"),
        (
            "Level 2",
            "Task 4",
            "Feature Engineering (6 features)",
            "04_Feature_Engineering.ipynb",
        ),
        ("Level 2", "Task 5", "SQL Analysis + Schema Design", "05_SQL_Analysis.ipynb"),
        (
            "Level 3",
            "Task 6",
            "End-to-End sklearn Pipeline",
            "06_End_to_End_Pipeline.ipynb",
        ),
        ("Level 3", "Task 7", "FastAPI Deployment", "api/app.py"),
        ("Level 4", "Task 8", "Case Study — Why Customers Churn", "Streamlit EDA page"),
        (
            "Level 4",
            "Task 9",
            "SHAP Model Explainability",
            "07_Model_Explainability.ipynb",
        ),
        (
            "Level 5",
            "Task 10",
            "Hybrid Recommendation System",
            "08_Recommendation_System.ipynb",
        ),
    ]

    df_tasks = pd.DataFrame(tasks, columns=["Level", "Task", "Description", "File"])
    st.dataframe(df_tasks, use_container_width=True, hide_index=True)

    st.markdown("### 🏗️ Architecture")
    st.code("""
ds_project/
├── 📁 data/                        ← telco_churn.csv + tmdb files
├── 📓 notebooks/                   ← 8 Jupyter notebooks (Tasks 1-10)
│   ├── 01_EDA_Data_Cleaning.ipynb
│   ├── 02_Basic_ML_Models.ipynb
│   ├── 03_Debug_Improve.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_SQL_Analysis.ipynb
│   ├── 06_End_to_End_Pipeline.ipynb
│   ├── 07_Model_Explainability.ipynb
│   └── 08_Recommendation_System.ipynb
├── ⚡ api/
│   └── app.py                      ← FastAPI REST endpoint
├── 🎈 streamlit_app/
│   └── app.py                      ← This Streamlit dashboard
├── 💾 models/
│   └── churn_pipeline.pkl          ← Trained model
├── 🗃️ sql/
│   └── queries.sql                 ← SQL queries file
└── 📄 README.md                    ← Setup & deployment guide
    """)
