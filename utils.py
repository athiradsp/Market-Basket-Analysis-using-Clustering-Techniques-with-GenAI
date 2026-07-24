"""
Utilities, Visualization Helpers & Viva Preparation Engine for Market Basket Analysis (CSM 355).

This module provides:
1. Interactive Plotly charts (RFM distributions, 3D Cluster Scatters, Elbow Curves, Rule Network Graphs).
2. Export helpers (TXT / Report download formatters).
3. Viva Voce Cheat Sheet & Quiz content tailored for Course CSM 355.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


# ---------------------------------------------------------
# VISUALIZATION FUNCTIONS (PLOTLY)
# ---------------------------------------------------------

def plot_rfm_distributions(rfm_raw: pd.DataFrame) -> go.Figure:
    """
    Creates subplots showing distributions of Recency, Frequency, and Monetary metrics.
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency Distribution (Days)", "Frequency Distribution (Orders)", "Monetary Distribution ($)")
    )

    fig.add_trace(
        go.Histogram(x=rfm_raw["Recency"], name="Recency", marker_color="#6366F1"),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=rfm_raw["Frequency"], name="Frequency", marker_color="#10B981"),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=rfm_raw["Monetary"], name="Monetary", marker_color="#F59E0B"),
        row=1, col=3
    )

    fig.update_layout(
        title_text="<b>RFM Metric Feature Distributions</b>",
        template="plotly_dark",
        showlegend=False,
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_top_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Creates an interactive bar chart of top selling products by order volume.
    """
    top_items = (
        df.groupby("Description")["Quantity"]
        .sum()
        .reset_index()
        .sort_values(by="Quantity", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        top_items,
        x="Quantity",
        y="Description",
        orientation="h",
        title=f"<b>Top {top_n} Most Purchased Retail Items</b>",
        labels={"Quantity": "Total Quantity Sold", "Description": "Product Description"},
        color="Quantity",
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_elbow_curve(wcss_list: List[float]) -> go.Figure:
    """
    Plots the K-Means WCSS (Inertia) Elbow curve to evaluate optimal cluster count k.
    """
    k_values = list(range(1, len(wcss_list) + 1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_values,
        y=wcss_list,
        mode="lines+markers",
        marker=dict(size=8, color="#EF4444"),
        line=dict(color="#EF4444", width=3),
        name="WCSS (Inertia)"
    ))

    fig.update_layout(
        title="<b>K-Means Elbow Method (WCSS vs Number of Clusters k)</b>",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Within-Cluster Sum of Squares (WCSS / Inertia)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_3d_clusters(rfm_scaled: pd.DataFrame, labels: np.ndarray, title: str = "3D RFM Customer Clusters") -> go.Figure:
    """
    Generates an interactive 3D Scatter Plot of RFM Customer Segments.
    """
    df_plot = rfm_scaled.copy()
    df_plot["Cluster"] = [f"Cluster {l}" if l != -1 else "Noise (-1)" for l in labels]

    fig = px.scatter_3d(
        df_plot,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Cluster",
        symbol="Cluster",
        title=f"<b>{title}</b>",
        labels={"Recency": "Recency (Std)", "Frequency": "Frequency (Std)", "Monetary": "Monetary (Std)"},
        opacity=0.85,
        template="plotly_dark"
    )
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def plot_rule_benchmark(benchmark_data: Dict[str, Any]) -> go.Figure:
    """
    Bar chart comparing runtime performance between Apriori and FP-Growth.
    """
    df_bm = pd.DataFrame([
        {"Algorithm": "Apriori", "Execution Time (Seconds)": benchmark_data["apriori_time_sec"]},
        {"Algorithm": "FP-Growth", "Execution Time (Seconds)": benchmark_data["fpgrowth_time_sec"]}
    ])

    fig = px.bar(
        df_bm,
        x="Algorithm",
        y="Execution Time (Seconds)",
        color="Algorithm",
        title="<b>Algorithm Performance Benchmark (Apriori vs FP-Growth Execution Time)</b>",
        text_auto=".4f",
        color_discrete_map={"Apriori": "#F59E0B", "FP-Growth": "#10B981"},
        template="plotly_dark"
    )
    fig.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_rules_scatter(rules_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of association rules: Support vs Confidence colored by Lift metric.
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No rules to display", template="plotly_dark")
        return fig

    fig = px.scatter(
        rules_df,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_data=["rule"],
        title="<b>Association Rules Metric Map (Support vs Confidence vs Lift)</b>",
        labels={"support": "Support Metric", "confidence": "Confidence Metric", "lift": "Lift Factor"},
        color_continuous_scale="Plasma",
        template="plotly_dark"
    )
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_network_graph(rules_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Generates an interactive 2D Network Graph of Item Association Rules using NetworkX & Plotly.
    Nodes represent Antecedents & Consequents; Edges represent directed rule association weighted by Lift.
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No rules available for Network Graph", template="plotly_dark")
        return fig

    top_rules = rules_df.sort_values(by="lift", ascending=False).head(top_n)

    G = nx.DiGraph()

    for _, row in top_rules.iterrows():
        ant = row.get("antecedents_str", "")
        seq = row.get("consequents_str", "")
        lift = row.get("lift", 1.0)
        G.add_edge(ant, seq, weight=lift)

    pos = nx.spring_layout(G, k=0.5, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = []
    node_y = []
    node_text = []
    node_degree = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Product Item: {node}<br>Connections: {G.degree(node)}")
        node_degree.append(G.degree(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[node for node in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_degree,
            size=[15 + deg * 5 for deg in node_degree],
            colorbar=dict(title="Node Degree"),
            line_width=2
        ),
        hovertext=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"<b>Interactive Product Association Network Graph (Top {len(top_rules)} Rules)</b>",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        height=500
    )
    return fig


# ---------------------------------------------------------
# VIVA VOCE EXAMINATION CHEAT SHEET CONTENT (CSM 355)
# ---------------------------------------------------------

VIVA_QUESTIONS = [
    {
        "category": "Market Basket Analysis & Association Rules",
        "question": "What is the mathematical definition of Support, Confidence, and Lift?",
        "answer": (
            "1. Support(A -> B) = P(A ∩ B) = Count(Transactions containing A and B) / Total Transactions.\n"
            "2. Confidence(A -> B) = P(B | A) = Support(A ∩ B) / Support(A).\n"
            "3. Lift(A -> B) = Confidence(A -> B) / Support(B) = P(A ∩ B) / (P(A) * P(B)).\n"
            "Interpretation of Lift: If Lift > 1, items A and B are positively correlated (co-occur more than by random chance). "
            "If Lift = 1, independent. If Lift < 1, negatively correlated (substitutes)."
        )
    },
    {
        "category": "Association Algorithms",
        "question": "How does FP-Growth differ from the Apriori algorithm, and why is FP-Growth faster?",
        "answer": (
            "Apriori generates candidate k-itemsets level-by-level and performs repeated costly database scans for each candidate length k. "
            "In contrast, FP-Growth compresses the transactional database into a compact Frequent Pattern Tree (FP-Tree) structure "
            "in just 2 database scans and mines frequent patterns directly without generating candidate itemsets. "
            "Hence, FP-Growth is significantly faster, especially on dense datasets with low min_support."
        )
    },
    {
        "category": "Customer Clustering",
        "question": "Explain Silhouette Score and Davies-Bouldin Index mathematically.",
        "answer": (
            "Silhouette Score S(i) = [b(i) - a(i)] / max(a(i), b(i)), where a(i) is mean intra-cluster distance and b(i) is mean nearest-cluster distance. "
            "Score ranges from -1 to +1 (higher is better).\n"
            "Davies-Bouldin Index (DBI) = (1/k) * sum(max_{j != i} ((sigma_i + sigma_j) / d(c_i, c_j))). "
            "Lower DBI indicates better separation and higher cluster compactness (0 is ideal)."
        )
    },
    {
        "category": "Unsupervised Learning",
        "question": "What is the difference between K-Means, DBSCAN, and Agglomerative Hierarchical Clustering?",
        "answer": (
            "1. K-Means: Centroid-based partitioning algorithm. Requires pre-specifying k. Assumes spherical clusters of equal variance.\n"
            "2. DBSCAN: Density-based algorithm. Does not require specifying k. Finds arbitrary-shaped clusters and identifies noise (-1).\n"
            "3. Agglomerative: Hierarchical bottom-up tree building. Uses linkage criteria (Ward, Complete, Average) to merge clusters."
        )
    },
    {
        "category": "GenAI Integration",
        "question": "How is Generative AI connected to Machine Learning output in this architecture?",
        "answer": (
            "Machine Learning algorithms perform numerical clustering (RFM centroids) and pattern mining (Lift rules). "
            "These quantitative outputs are formatted into a clean JSON Context Payload. "
            "The JSON payload is passed with a role-based System Prompt to an LLM (OpenAI/Anthropic/Ollama), "
            "which synthesizes the numbers into human-readable business strategy, customer personas, and inventory recommendations."
        )
    }
]
