"""
Vercel Serverless Entry Point for Market Basket Analysis & GenAI (CSM 355).

This module provides a serverless Flask HTTP bridge to execute data processing,
customer clustering, association rule mining, and GenAI report generation on Vercel.
"""

from flask import Flask, jsonify, request
import os
import sys

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loader
import clustering
import association_rules
import genai_insights

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "project": "Market Basket Analysis using Clustering Techniques with GenAI",
        "message": "Streamlit frontend deployed. Serverless ML API active."
    })

@app.route('/favicon.ico', methods=['GET'])
@app.route('/favicon.png', methods=['GET'])
def favicon():
    return '', 204

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        req_data = request.get_json(silent=True) or {}
        num_records = req_data.get('num_records', 1000)
        n_clusters = req_data.get('n_clusters', 3)
        min_support = req_data.get('min_support', 0.03)

        # Execution Pipeline
        df_raw = data_loader.generate_synthetic_data(num_records=num_records)
        df_clean, audit = data_loader.clean_data(df_raw)
        rfm_raw, rfm_scaled = data_loader.compute_rfm(df_clean)
        km, labels, centroids, wcss = clustering.run_kmeans(rfm_scaled, n_clusters=n_clusters)
        eval_metrics = clustering.calculate_cluster_metrics(rfm_scaled, labels)
        basket = data_loader.create_basket_matrix(df_clean)
        fp_itemsets, fp_rules = association_rules.mine_fpgrowth_rules(basket, min_support=min_support)

        rules_list = []
        if not fp_rules.empty:
            for _, r in fp_rules.head(5).iterrows():
                rules_list.append({
                    "rule": r.get("rule", ""),
                    "support": float(r.get("support", 0)),
                    "confidence": float(r.get("confidence", 0)),
                    "lift": float(r.get("lift", 0))
                })

        return jsonify({
            "status": "success",
            "cleaned_rows": len(df_clean),
            "metrics": eval_metrics,
            "mined_rules_count": len(fp_rules),
            "top_rules": rules_list
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
