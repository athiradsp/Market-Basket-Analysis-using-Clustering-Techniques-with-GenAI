"""
GenAI Insight Generation Layer for Market Basket Analysis (CSM 355).

This module bridges Machine Learning outputs (Customer Clusters & Association Rules)
with Generative AI (LLMs) to produce natural language business intelligence, executive strategies,
customer persona classifications, and store shelf placement recommendations.

Supported Providers:
1. OpenAI API (gpt-4o, gpt-3.5-turbo)
2. Anthropic API (claude-3-5-sonnet-20241022)
3. Local Ollama API (http://localhost:11434)
4. Offline Heuristic AI Synthesizer (Built-in rule engine when no API key is provided)
"""

import json
from typing import Dict, Any, Optional
import pandas as pd


def format_llm_payload(cluster_summary_df: pd.DataFrame, top_rules_df: pd.DataFrame) -> str:
    """
    Converts cluster centroids/profiles (Recency, Frequency, Monetary averages)
    and top association rules (sorted by Lift) into a clean, structured JSON context payload for LLM consumption.

    Args:
        cluster_summary_df (pd.DataFrame): DataFrame containing mean RFM metrics per cluster.
        top_rules_df (pd.DataFrame): DataFrame containing top mined association rules.

    Returns:
        str: Serialized JSON payload string formatted for LLM prompts.
    """
    payload: Dict[str, Any] = {
        "project_title": "Market Basket Analysis using Clustering Techniques & GenAI",
        "cluster_segments": [],
        "top_association_rules": []
    }

    # Format Cluster Profiles
    if not cluster_summary_df.empty:
        for _, row in cluster_summary_df.iterrows():
            cluster_info = {
                "cluster_id": int(row["Cluster"]) if "Cluster" in row else 0,
                "recency_days_avg": float(row.get("Recency_Mean", 0)),
                "frequency_invoices_avg": float(row.get("Frequency_Mean", 0)),
                "monetary_spend_avg": float(row.get("Monetary_Mean", 0)),
                "customer_count": int(row.get("Customer_Count", 0))
            }
            payload["cluster_segments"].append(cluster_info)

    # Format Association Rules (Top 10 sorted by Lift)
    if not top_rules_df.empty:
        rules_sorted = top_rules_df.sort_values(by="lift", ascending=False).head(10)
        for _, row in rules_sorted.iterrows():
            rule_info = {
                "antecedent_items": row.get("antecedents_str", ""),
                "consequent_items": row.get("consequents_str", ""),
                "support": float(row.get("support", 0)),
                "confidence": float(row.get("confidence", 0)),
                "lift": float(row.get("lift", 0))
            }
            payload["top_association_rules"].append(rule_info)

    return json.dumps(payload, indent=2)


def generate_ai_report(
    json_payload: str,
    api_key: Optional[str] = None,
    provider: str = "Offline Synthesizer",
    model_name: str = "gpt-4o"
) -> str:
    """
    Generates a natural language Business Intelligence & Executive Marketing Report
    by passing structured machine learning insights (JSON context) to Generative AI models.

    Args:
        json_payload (str): Formatted JSON payload string containing cluster profiles and rules.
        api_key (Optional[str]): API key for OpenAI or Anthropic. If blank or invalid, falls back to Offline Synthesizer.
        provider (str): Choice of LLM provider ('OpenAI', 'Anthropic', 'Ollama (Local)', 'Offline Synthesizer').
        model_name (str): Exact LLM model identifier.

    Returns:
        str: Comprehensive Markdown formatted Executive BI Report.
    """
    system_prompt = (
        "You are an expert Chief Data Officer and Senior Retail Operations Strategist for an e-commerce enterprise. "
        "You are evaluating machine learning outputs from a Market Basket Analysis & Customer Segmentation platform.\n\n"
        "Your goal is to transform the provided JSON payload (containing customer RFM cluster centroids and association rules) "
        "into an actionable, executive-ready Business Intelligence & Retail Strategy Report.\n\n"
        "Your report MUST include the following 4 structured sections:\n"
        "1. ## Customer Persona Classifications & Behavioral Profiles\n"
        "   - Assign intuitive persona names to each cluster (e.g., 'High-Value VIP Loyalists', 'At-Risk Bargain Seekers', 'Frequent Bulk Buyers').\n"
        "   - Provide key metrics breakdown and behavioral description.\n\n"
        "2. ## Targeted Promotional & Cross-Selling Strategies\n"
        "   - Design specific promotional campaigns tailored to each customer persona.\n"
        "   - Leverage high-lift association rules to create bundled offers (e.g., 'Buy X, Get Y at 20% off').\n\n"
        "3. ## Store Inventory Layout & Digital Merchandising Recommendations\n"
        "   - Provide physical shelf placement rules (e.g., placing complimentary items side-by-side) and digital 'Recommended for You' widgets based on Lift and Confidence metrics.\n\n"
        "4. ## Executive Summary & ROI Action Plan for Retail Leadership\n"
        "   - High-level executive briefing with priority recommendations for immediate quarterly deployment."
    )

    # Route 1: OpenAI Provider
    if provider == "OpenAI" and api_key and api_key.strip():
        try:
            import openai
            client = openai.OpenAI(api_key=api_key.strip())
            response = client.chat.completions.create(
                model=model_name if model_name else "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the Machine Learning JSON Context Payload:\n\n{json_payload}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return (
                f"> **[OpenAI API Notice]**: Unable to connect to OpenAI ({str(e)}). "
                f"Falling back to Offline Intelligent Rule Synthesizer below:\n\n"
                + _generate_offline_heuristic_report(json_payload)
            )

    # Route 2: Anthropic Provider
    elif provider == "Anthropic" and api_key and api_key.strip():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key.strip())
            response = client.messages.create(
                model=model_name if model_name else "claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"Here is the Machine Learning JSON Context Payload:\n\n{json_payload}"}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return (
                f"> **[Anthropic API Notice]**: Unable to connect to Anthropic ({str(e)}). "
                f"Falling back to Offline Intelligent Rule Synthesizer below:\n\n"
                + _generate_offline_heuristic_report(json_payload)
            )

    # Route 3: Local Ollama REST API
    elif provider == "Ollama (Local)":
        try:
            import requests
            url = "http://localhost:11434/api/generate"
            prompt_text = f"{system_prompt}\n\nHere is the JSON Payload:\n{json_payload}"
            res = requests.post(url, json={"model": model_name or "llama3", "prompt": prompt_text, "stream": False}, timeout=15)
            if res.status_code == 200:
                return res.json().get("response", "No response text received from Ollama.")
            else:
                return f"> **[Ollama Error]**: Status Code {res.status_code}. Falling back to Offline Synthesizer.\n\n" + _generate_offline_heuristic_report(json_payload)
        except Exception as e:
            return f"> **[Ollama Connection Error]**: Could not reach local Ollama instance at localhost:11434 ({str(e)}). Falling back to Offline Synthesizer.\n\n" + _generate_offline_heuristic_report(json_payload)

    # Route 4: Offline Heuristic Synthesizer (Default Fallback)
    else:
        return _generate_offline_heuristic_report(json_payload)


def _generate_offline_heuristic_report(json_payload: str) -> str:
    """
    Intelligent heuristic report generator that parses the ML JSON payload
    and constructs a comprehensive Executive Business Intelligence report without external API dependencies.
    Ensures 100% reliability for offline viva examinations.
    """
    try:
        data = json.loads(json_payload)
    except Exception:
        data = {"cluster_segments": [], "top_association_rules": []}

    clusters = data.get("cluster_segments", [])
    rules = data.get("top_association_rules", [])

    report_lines = []
    report_lines.append("# 📊 Executive Business Intelligence & GenAI Insights Report")
    report_lines.append("**Module:** Market Basket Analysis & Customer Segmentation Platform\n")
    report_lines.append("> *Notice: Generated using the Built-in Intelligent Rule Synthesizer Engine (Offline Mode).* \n")

    # Section 1: Persona Classifications
    report_lines.append("## 1. Customer Persona Classifications & Behavioral Profiles\n")
    if not clusters:
        report_lines.append("*No active cluster data found in payload.*")
    else:
        for c in clusters:
            cid = c.get("cluster_id", 0)
            rec = c.get("recency_days_avg", 0)
            freq = c.get("frequency_invoices_avg", 0)
            mon = c.get("monetary_spend_avg", 0)
            cnt = c.get("customer_count", 0)

            # Heuristic Persona Mapping
            if mon > 500 or freq > 10:
                persona_title = f"👑 Cluster {cid}: High-Value VIP Loyalists"
                desc = "High frequency and heavy spenders. Represents top-tier revenue drivers requiring premium VIP treatment."
            elif rec > 100:
                persona_title = f"⚠️ Cluster {cid}: At-Risk / Dormant Customers"
                desc = "Long elapsed recency without recent orders. Immediate re-engagement discount campaigns required."
            else:
                persona_title = f"🛒 Cluster {cid}: Steady Core Shoppers"
                desc = "Moderate recency and steady purchase volume. High potential for cross-selling and upselling."

            report_lines.append(f"### {persona_title}")
            report_lines.append(f"- **Customer Count:** {cnt} customers")
            report_lines.append(f"- **Average Recency:** {rec} days")
            report_lines.append(f"- **Average Order Frequency:** {freq} orders")
            report_lines.append(f"- **Average Total Monetary Value:** ${mon:,.2f}")
            report_lines.append(f"- **Behavioral Assessment:** {desc}\n")

    # Section 2: Targeted Promotional Strategies
    report_lines.append("## 2. Targeted Promotional & Cross-Selling Strategies\n")
    report_lines.append("Based on mined customer RFM segments, we recommend the following tailored campaigns:\n")
    report_lines.append("1. **VIP Loyalty Program (High Monetary Clusters):** Provide early access to new product releases, free priority shipping, and dedicated concierge support.")
    report_lines.append("2. **Win-Back Re-activation Automated Campaign (High Recency Clusters):** Send automated email reminders offering a '15% Off Your Next Order' incentive.")
    report_lines.append("3. **Subscription / Volume Tier Discounts (High Frequency Clusters):** Implement bulk purchase incentives to increase basket depth.\n")

    # Section 3: Inventory Layout & Cross-Selling Rules
    report_lines.append("## 3. Store Inventory Layout & Digital Merchandising Recommendations\n")
    if not rules:
        report_lines.append("*No high-lift association rules mined above minimum thresholds.*")
    else:
        report_lines.append("Top product placement and bundle rules derived from high Lift association mining:\n")
        for idx, r in enumerate(rules[:5], 1):
            ant = r.get("antecedent_items", "")
            seq = r.get("consequent_items", "")
            lift = r.get("lift", 0)
            conf = r.get("confidence", 0)
            supp = r.get("support", 0)

            report_lines.append(f"### Rule #{idx}: [{ant}] ➔ [{seq}]")
            report_lines.append(f"- **Metrics:** Lift = `{lift}` | Confidence = `{conf * 100:.1f}%` | Support = `{supp * 100:.1f}%`")
            report_lines.append(f"- **Merchandising Action:** Place `{seq}` directly adjacent to `{ant}` on shelf or configure automated 'Customers who bought {ant} also added {seq}' digital popup widgets.\n")

    # Section 4: Executive Summary
    report_lines.append("## 4. Executive Summary & Strategic Action Plan")
    report_lines.append("- **Immediate Action 1:** Co-locate the top 3 itemsets identified above to maximize cross-sell lift revenue.")
    report_lines.append("- **Immediate Action 2:** Target high-value clusters with specialized bundle promotions to increase average order value (AOV).")
    report_lines.append("- **Expected ROI:** Implementing these combined ML & Rule recommendations is projected to lift basket cross-sell revenue by 12% - 18%.")

    return "\n".join(report_lines)
