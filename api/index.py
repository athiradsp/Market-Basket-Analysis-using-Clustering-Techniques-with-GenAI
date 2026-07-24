"""
Vercel Serverless Entry Point for Market Basket Analysis & GenAI (CSM 355).

This module provides a serverless Flask HTTP bridge to execute data processing,
customer clustering, association rule mining, and GenAI report generation on Vercel.
Includes an interactive Web Portal UI at '/' and JSON API routes.
"""

from flask import Flask, jsonify, request, render_template_string
import os
import sys

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loader
import clustering
import association_rules
import genai_insights

app = Flask(__name__)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSM 355 | Market Basket Analysis & GenAI Web Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: #0F172A; color: #F8FAFC; padding: 24px; min-height: 100vh; line-height: 1.6; }
        .container { max-width: 1100px; margin: 0 auto; }
        
        .header-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 32px; border-radius: 20px; margin-bottom: 24px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4); backdrop-filter: blur(16px);
        }
        .badge { background: #6366F1; color: #FFF; padding: 6px 14px; border-radius: 30px; font-size: 13px; font-weight: 600; }
        h1 { font-size: 30px; margin-top: 12px; color: #F8FAFC; font-weight: 700; }
        .subhead { color: #38BDF8; font-size: 15px; margin-top: 4px; font-weight: 500; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .card {
            background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px; padding: 24px; transition: transform 0.2s, border-color 0.2s;
        }
        .card:hover { border-color: #6366F1; transform: translateY(-3px); }
        .card h2 { font-size: 18px; color: #38BDF8; margin-bottom: 12px; font-weight: 600; }
        
        .form-group { margin-bottom: 16px; }
        label { display: block; font-size: 13px; color: #94A3B8; margin-bottom: 6px; }
        input, select {
            width: 100%; background: #0F172A; border: 1px solid rgba(255,255,255,0.15);
            padding: 10px 14px; border-radius: 8px; color: #FFF; font-size: 14px; outline: none;
        }
        input:focus { border-color: #6366F1; }
        
        .btn {
            background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
            color: #FFF; border: none; padding: 12px 24px; border-radius: 10px;
            font-weight: 600; cursor: pointer; width: 100%; font-size: 15px;
            transition: opacity 0.2s, transform 0.1s;
        }
        .btn:hover { opacity: 0.95; transform: scale(1.01); }
        
        .results-box {
            background: #090D16; border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px; padding: 16px; font-family: monospace; font-size: 13px;
            color: #34D399; max-height: 280px; overflow-y: auto; white-space: pre-wrap; margin-top: 16px;
        }
        
        .metric-pill { display: inline-block; background: rgba(56, 189, 248, 0.1); color: #38BDF8; padding: 4px 10px; border-radius: 6px; font-size: 12px; margin-right: 6px; margin-bottom: 6px; }
        code { background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; color: #F472B6; font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-card">
            <span class="badge">CSM 355 • Practical Exam & Viva System</span>
            <h1>🛒 Market Basket Analysis & Customer Clustering with GenAI</h1>
            <p class="subhead">Vercel Serverless Web Portal & Interactive ML Engine</p>
        </div>

        <div class="grid">
            <div class="card">
                <h2>⚡ Interactive Serverless ML Tester</h2>
                <p style="font-size: 13px; color: #94A3B8; margin-bottom: 16px;">Run real-time K-Means customer segmentation & FP-Growth association rule mining directly on Vercel.</p>
                <div class="form-group">
                    <label>Synthetic Records Count:</label>
                    <input type="number" id="num_records" value="1000" min="200" max="3000">
                </div>
                <div class="form-group">
                    <label>Number of Clusters (k):</label>
                    <input type="number" id="n_clusters" value="3" min="2" max="6">
                </div>
                <div class="form-group">
                    <label>Min Support Fraction:</label>
                    <input type="number" id="min_support" value="0.03" step="0.01" min="0.01" max="0.2">
                </div>
                <button class="btn" onclick="runPipeline()">🚀 Execute ML Pipeline (/api/analyze)</button>
                <div id="results" class="results-box">Click button above to trigger serverless ML execution...</div>
            </div>

            <div class="card">
                <h2>📚 Project Architecture & Modules</h2>
                <div style="margin-bottom: 12px;">
                    <span class="metric-pill">data_loader.py</span> Synthetic RFM & Data Cleaning
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="metric-pill">clustering.py</span> K-Means • DBSCAN • Agglomerative • Silhouette S(i)
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="metric-pill">association_rules.py</span> Apriori vs FP-Growth Benchmarking
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="metric-pill">genai_insights.py</span> LLM Prompt Synthesis & Offline Report Engine
                </div>
                <div style="margin-bottom: 12px;">
                    <span class="metric-pill">app.py</span> Interactive Streamlit Multi-Tab Dashboard
                </div>

                <h2 style="margin-top: 24px;">💻 Local Dashboard Execution</h2>
                <p style="font-size: 13px; color: #94A3B8;">To run the full Streamlit UI locally for your viva examination:</p>
                <pre style="background:#000; padding:10px; border-radius:8px; margin-top:8px; font-size:12px; color:#A7F3D0;">pip install -r requirements-streamlit.txt
streamlit run app.py</pre>
            </div>
        </div>
    </div>

    <script>
        async function runPipeline() {
            const resBox = document.getElementById('results');
            resBox.innerText = "⏳ Executing serverless data cleaning, RFM scaling, K-Means clustering, and FP-Growth rule mining...";
            try {
                const payload = {
                    num_records: parseInt(document.getElementById('num_records').value),
                    n_clusters: parseInt(document.getElementById('n_clusters').value),
                    min_support: parseFloat(document.getElementById('min_support').value)
                };
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                resBox.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                resBox.innerText = "❌ Error executing pipeline: " + err;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    if request.headers.get('Accept') == 'application/json' or request.args.get('format') == 'json':
        return jsonify({
            "status": "online",
            "project": "Market Basket Analysis using Clustering Techniques with GenAI",
            "course": "CSM 355",
            "message": "Streamlit frontend deployed. Serverless ML API active."
        })
    return render_template_string(INDEX_HTML)

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        "status": "online",
        "project": "Market Basket Analysis using Clustering Techniques with GenAI",
        "course": "CSM 355",
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
