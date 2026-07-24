"""
Main Streamlit Application: Market Basket Analysis using Clustering Techniques with GenAI
Course Code: CSM 355 | End-Semester Project & Viva Voce Demonstration System

This application integrates:
- Tab 1: Data Preprocessing & Interactive EDA
- Tab 2: Customer Clustering (K-Means, DBSCAN, Agglomerative) & Metrics (Silhouette, DBI)
- Tab 3: Association Rule Mining (Apriori vs FP-Growth Benchmark, Network Graph)
- Tab 4: GenAI Business Intelligence Report (OpenAI, Anthropic, Ollama, Offline Fallback)
- Tab 5: Viva Voce Prep & Mathematical Formula Cheat Sheet (CSM 355)
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import custom application modules
import data_loader
import clustering
import association_rules
import genai_insights
import utils

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="CSM 355 | Market Basket Analysis & GenAI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Main Background & Font Styling */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
    }
    
    /* Header Card Styling */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 24px;
    }
    
    /* Metric KPI Cards */
    .metric-card {
        background: rgba(51, 65, 85, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #6366F1;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# MAIN HEADER BANNER
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 28px; color: #F8FAFC;">🛒 Market Basket Analysis with Clustering & GenAI</h1>
            <p style="margin: 4px 0 0 0; color: #38BDF8; font-size: 15px; font-weight: 500;">
                Course Code: CSM 355 | Practical Examination & Viva Voce Demonstration
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: #6366F1; color: #FFF; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                Tech Stack: Streamlit • Scikit-Learn • MLxtend • GenAI
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR CONTROLS & HYPERPARAMETERS
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Data & Control Center")
    
    # 1. Dataset Selection
    st.subheader("1. Dataset Source")
    data_source = st.radio(
        "Select Data Input Method:",
        options=["Use Synthetic Dataset (Built-in)", "Upload CSV File"],
        index=0
    )
    
    df_raw = pd.DataFrame()
    
    if data_source == "Use Synthetic Dataset (Built-in)":
        num_records = st.slider("Synthetic Transactions Count:", min_value=500, max_value=5000, value=1500, step=250)
        df_raw = data_loader.generate_synthetic_data(num_records=num_records)
    else:
        uploaded_file = st.file_uploader("Upload Retail CSV Dataset", type=["csv"])
        
        # Download Sample Template CSV Button
        sample_df = data_loader.generate_synthetic_data(num_records=100)
        sample_csv_data = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_csv_data,
            file_name="sample_retail_data.csv",
            mime="text/csv",
            help="Click to download a valid sample CSV dataset template to test uploading."
        )

        if uploaded_file is not None:
            try:
                # Robust multi-encoding reader
                try:
                    df_raw = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, encoding="latin1")
                except Exception:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
                
                st.success(f"✅ Uploaded CSV loaded ({len(df_raw)} rows, {len(df_raw.columns)} columns).")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.info("Using fallback synthetic dataset in the interim.")
                df_raw = data_loader.generate_synthetic_data(num_records=1000)
        else:
            st.info("Awaiting file upload. Using built-in synthetic data in the interim.")
            df_raw = data_loader.generate_synthetic_data(num_records=1000)

    st.divider()

    # 2. Clustering Parameters
    st.subheader("2. Customer Clustering")
    cluster_algorithm = st.selectbox(
        "Clustering Model:",
        options=["K-Means", "DBSCAN", "Agglomerative (Hierarchical)"]
    )

    if cluster_algorithm == "K-Means":
        kmeans_k = st.slider("Number of Clusters (k):", min_value=2, max_value=8, value=3)
    elif cluster_algorithm == "DBSCAN":
        dbscan_eps = st.slider("DBSCAN Epsilon (eps):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        dbscan_min_samples = st.slider("Min Samples:", min_value=2, max_value=15, value=5)
    else:
        agg_k = st.slider("Agglomerative Clusters:", min_value=2, max_value=8, value=3)
        agg_linkage = st.selectbox("Linkage Type:", options=["ward", "complete", "average"])

    st.divider()

    # 3. Association Mining Parameters
    st.subheader("3. Association Mining")
    min_support = st.slider("Min Support (Fraction):", min_value=0.01, max_value=0.20, value=0.03, step=0.01)
    min_confidence = st.slider("Min Confidence (Fraction):", min_value=0.10, max_value=0.90, value=0.20, step=0.05)
    min_lift_filter = st.slider("Min Lift Filter:", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

    st.divider()

    # 4. GenAI Settings
    st.subheader("4. GenAI Engine Settings")
    ai_provider = st.selectbox(
        "Select LLM Provider:",
        options=["Offline Synthesizer", "OpenAI", "Anthropic", "Ollama (Local)"],
        index=0
    )
    
    api_key_input = ""
    model_name_input = ""
    
    if ai_provider in ["OpenAI", "Anthropic"]:
        api_key_input = st.text_input(f"{ai_provider} API Key:", type="password", help="Leave empty to use Offline Fallback.")
        if ai_provider == "OpenAI":
            model_name_input = st.selectbox("Model:", ["gpt-4o", "gpt-3.5-turbo"])
        else:
            model_name_input = st.selectbox("Model:", ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"])
    elif ai_provider == "Ollama (Local)":
        model_name_input = st.text_input("Ollama Model Name:", value="llama3")


# ---------------------------------------------------------
# DATA PIPELINE EXECUTION
# ---------------------------------------------------------
try:
    # Step 1: Clean Data & Validate Schema
    df_cleaned, audit_stats = data_loader.clean_data(df_raw)
except Exception as e:
    st.error(f"⚠️ CSV Validation Notice: {e}")
    st.info("💡 Tip: Use the 'Download Sample CSV Template' button in the sidebar to get a ready-to-use valid dataset.")
    df_raw = data_loader.generate_synthetic_data(num_records=1000)
    df_cleaned, audit_stats = data_loader.clean_data(df_raw)

# Step 2: Compute RFM Metrics
rfm_raw, rfm_scaled = data_loader.compute_rfm(df_cleaned)

# Step 3: Compute Basket Matrix
basket_matrix = data_loader.create_basket_matrix(df_cleaned)


# ---------------------------------------------------------
# INTERACTIVE DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Data Preprocessing & EDA",
    "🎯 2. Customer Clustering",
    "🛒 3. Association Mining (Apriori vs FP)",
    "🤖 4. GenAI BI Report Generator",
    "🎓 5. Viva Voce Examination Guide"
])


# =========================================================
# TAB 1: DATA PREPROCESSING & EDA
# =========================================================
with tab1:
    st.subheader("📋 Dataset Overview & Cleaning Pipeline Audit")
    
    # Audit KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{audit_stats["initial_rows"]}</div><div class="metric-lbl">Raw Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{audit_stats["cleaned_rows"]}</div><div class="metric-lbl">Cleaned Rows</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{audit_stats["canceled_invoices"]}</div><div class="metric-lbl">Canceled/Returns Dropped</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{audit_stats["missing_customer_ids"]}</div><div class="metric-lbl">Null Customers Dropped</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{audit_stats["retention_rate_pct"]}%</div><div class="metric-lbl">Data Retention Rate</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations
    col_a, col_b = st.columns(2)
    with col_a:
        if not rfm_raw.empty:
            fig_rfm = utils.plot_rfm_distributions(rfm_raw)
            st.plotly_chart(fig_rfm, use_container_width=True)
    with col_b:
        if not df_cleaned.empty:
            fig_top = utils.plot_top_products(df_cleaned, top_n=10)
            st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("🔍 Cleaned Dataset Preview (First 5 Rows)")
    st.dataframe(df_cleaned.head(5), use_container_width=True)


# =========================================================
# TAB 2: CUSTOMER CLUSTERING & SEGMENTATION
# =========================================================
with tab2:
    st.subheader(f"🎯 Customer Segmentation using {cluster_algorithm}")

    if rfm_scaled.empty:
        st.warning("RFM Dataset is empty. Cannot perform clustering.")
    else:
        labels = np.array([])
        centroids = pd.DataFrame()
        wcss_list = []

        # Run Selected Model
        if cluster_algorithm == "K-Means":
            km_model, labels, centroids, wcss_list = clustering.run_kmeans(rfm_scaled, n_clusters=kmeans_k)
        elif cluster_algorithm == "DBSCAN":
            db_model, labels = clustering.run_dbscan(rfm_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples)
        else:
            agg_model, labels = clustering.run_hierarchical(rfm_scaled, n_clusters=agg_k, linkage=agg_linkage)

        # Calculate Evaluation Metrics
        eval_metrics = clustering.calculate_cluster_metrics(rfm_scaled, labels)

        # Display Evaluation Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            sil_str = f"{eval_metrics['silhouette_score']}" if eval_metrics['silhouette_score'] is not None else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-val">{sil_str}</div><div class="metric-lbl">Silhouette Score S(i)</div></div>', unsafe_allow_html=True)
        with col_m2:
            dbi_str = f"{eval_metrics['davies_bouldin_index']}" if eval_metrics['davies_bouldin_index'] is not None else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-val">{dbi_str}</div><div class="metric-lbl">Davies-Bouldin Index (DBI)</div></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{eval_metrics["n_clusters"]}</div><div class="metric-lbl">Valid Clusters</div></div>', unsafe_allow_html=True)
        with col_m4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{eval_metrics["noise_count"]}</div><div class="metric-lbl">Noise Points (-1)</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotting Layout
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_3d = utils.plot_3d_clusters(rfm_scaled, labels, title=f"3D RFM Clusters ({cluster_algorithm})")
            st.plotly_chart(fig_3d, use_container_width=True)
        with col_c2:
            if cluster_algorithm == "K-Means" and wcss_list:
                fig_elbow = utils.plot_elbow_curve(wcss_list)
                st.plotly_chart(fig_elbow, use_container_width=True)
            else:
                st.info("Elbow Curve analysis is specific to K-Means WCSS optimization.")

        # Cluster Profiles Summary
        st.subheader("📊 Customer Cluster Behavioral Profiles (Mean RFM Values)")
        profile_df = clustering.compute_cluster_profiles(rfm_raw, labels)
        st.dataframe(profile_df, use_container_width=True)


# =========================================================
# TAB 3: ASSOCIATION RULE MINING
# =========================================================
with tab3:
    st.subheader("🛒 Market Basket Association Rule Mining")

    if basket_matrix.empty:
        st.warning("Basket Matrix is empty. Unable to mine association rules.")
    else:
        # Mine Rules using both algorithms
        ap_itemsets, ap_rules = association_rules.mine_apriori_rules(
            basket_matrix, min_support=min_support, min_threshold=min_confidence
        )
        fp_itemsets, fp_rules = association_rules.mine_fpgrowth_rules(
            basket_matrix, min_support=min_support, min_threshold=min_confidence
        )

        # Execute Algorithm Benchmark
        benchmark_results = association_rules.benchmark_algorithms(basket_matrix, min_support=min_support)

        # Benchmark Cards
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{benchmark_results["apriori_time_sec"]}s</div><div class="metric-lbl">Apriori Time</div></div>', unsafe_allow_html=True)
        with b_col2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{benchmark_results["fpgrowth_time_sec"]}s</div><div class="metric-lbl">FP-Growth Time</div></div>', unsafe_allow_html=True)
        with b_col3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{benchmark_results["speedup_factor"]}x</div><div class="metric-lbl">FP-Growth Speedup</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Benchmarking & Scatter plot
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_bm = utils.plot_rule_benchmark(benchmark_results)
            st.plotly_chart(fig_bm, use_container_width=True)
        with col_r2:
            # Filter FP-Growth rules by user lift slider
            active_rules = fp_rules[fp_rules["lift"] >= min_lift_filter] if not fp_rules.empty else pd.DataFrame()
            fig_scatter = utils.plot_rules_scatter(active_rules)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Network Graph
        st.subheader("🕸️ Product Association Network Graph")
        fig_net = utils.plot_network_graph(active_rules, top_n=15)
        st.plotly_chart(fig_net, use_container_width=True)

        # Interactive Rules Dataframe
        st.subheader(f"📜 Mined Association Rules (Filtered by Lift >= {min_lift_filter})")
        if not active_rules.empty:
            display_cols = ["rule", "antecedents_str", "consequents_str", "support", "confidence", "lift"]
            st.dataframe(
                active_rules[display_cols].sort_values(by="lift", ascending=False),
                use_container_width=True
            )
        else:
            st.info("No rules found matching current minimum support, confidence, and lift thresholds. Try lowering sliders in the sidebar.")


# =========================================================
# TAB 4: GENAI BUSINESS INTELLIGENCE REPORT
# =========================================================
with tab4:
    st.subheader("🤖 Generative AI Business Intelligence Report Synthesis")

    # Generate JSON Context Payload
    current_profile = clustering.compute_cluster_profiles(rfm_raw, labels) if 'labels' in locals() else pd.DataFrame()
    current_rules = fp_rules if 'fp_rules' in locals() else pd.DataFrame()
    
    json_payload_str = genai_insights.format_llm_payload(current_profile, current_rules)

    with st.expander("🔍 Inspect JSON Context Payload passed to LLM"):
        st.code(json_payload_str, language="json")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Generate AI Executive Report", type="primary"):
        with st.spinner(f"Synthesizing Executive Report using {ai_provider}..."):
            report_markdown = genai_insights.generate_ai_report(
                json_payload=json_payload_str,
                api_key=api_key_input,
                provider=ai_provider,
                model_name=model_name_input
            )
            st.session_state["genai_report"] = report_markdown

    if "genai_report" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["genai_report"])

        st.download_button(
            label="📥 Download Executive Report (.txt)",
            data=st.session_state["genai_report"],
            file_name="CSM355_GenAI_Market_Basket_Report.txt",
            mime="text/plain"
        )


# =========================================================
# TAB 5: VIVA VOCE EXAMINATION GUIDE (CSM 355)
# =========================================================
with tab5:
    st.subheader("🎓 Course CSM 355: Viva Voce Examination Cheat Sheet & Theory Guide")
    st.info("Use this interactive reference during your practical examination & viva defense.")

    st.markdown("### 🧮 1. Mathematical Formulas & Core Metrics Cheat Sheet")
    
    st.latex(r"S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}")
    st.caption("Silhouette Score: Measures cluster cohesion vs separation. Range [-1, +1].")

    st.latex(r"DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)")
    st.caption("Davies-Bouldin Index: Ratio of intra-cluster distance to inter-cluster separation. Lower is better.")

    st.latex(r"\text{Support}(A \rightarrow B) = P(A \cap B) = \frac{\text{Transactions containing } A \text{ and } B}{\text{Total Transactions}}")
    st.latex(r"\text{Confidence}(A \rightarrow B) = P(B | A) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)}")
    st.latex(r"\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)} = \frac{P(A \cap B)}{P(A) \cdot P(B)}")

    st.markdown("---")
    st.markdown("### ❓ 2. Viva Voce Examiner Questions & Standard Answers")

    for idx, q_data in enumerate(utils.VIVA_QUESTIONS, 1):
        with st.expander(f"Q{idx} [{q_data['category']}]: {q_data['question']}"):
            st.markdown(f"**Standard Viva Response:**\n\n{q_data['answer']}")
