"""
Association Rule Mining Module for Market Basket Analysis (CSM 355).

This module implements two classic itemset mining algorithms:
1. Apriori Algorithm (Level-wise search using candidate generation)
2. FP-Growth Algorithm (Frequent Pattern Tree structure, array-based mining)

Mathematical Metrics Computed:
- Support(A -> B) = P(A union B) = Transactions containing both A and B / Total Transactions
- Confidence(A -> B) = P(B | A) = Support(A union B) / Support(A)
- Lift(A -> B) = Confidence(A -> B) / Support(B) = P(A union B) / (P(A) * P(B))
  - Lift > 1: Positive correlation (Items co-occur more than by chance)
  - Lift = 1: Independence
  - Lift < 1: Negative correlation (Items substitute each other)
"""

from typing import Tuple, Dict, Any
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def mine_apriori_rules(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.02,
    min_threshold: float = 0.2,
    metric: str = "confidence"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mines frequent itemsets and association rules using the Apriori Algorithm.

    Args:
        basket_matrix (pd.DataFrame): One-hot encoded transaction matrix (Invoices x Products).
        min_support (float): Minimum support threshold fraction [0.0, 1.0]. Default is 0.02.
        min_threshold (float): Minimum threshold for rule extraction metric. Default is 0.2.
        metric (str): Metric to filter rules ('confidence', 'lift', 'support'). Default is 'confidence'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - frequent_itemsets: DataFrame of itemsets meeting min_support.
            - rules: Formatted DataFrame of association rules with antecedents, consequents, support, confidence, lift.
    """
    if basket_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Step 1: Mine frequent itemsets via Apriori
    frequent_itemsets = apriori(
        basket_matrix,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["support", "itemsets"]), pd.DataFrame()

    # Step 2: Generate association rules
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold
    )

    if not rules.empty:
        # Convert frozensets to readable string lists
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules["rule"] = rules["antecedents_str"] + " ➔ " + rules["consequents_str"]
        
        # Round metric columns
        for col in ["support", "confidence", "lift", "leverage", "conviction"]:
            if col in rules.columns:
                rules[col] = rules[col].round(4)

    return frequent_itemsets, rules


def mine_fpgrowth_rules(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.02,
    min_threshold: float = 0.2,
    metric: str = "confidence"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mines frequent itemsets and association rules using the FP-Growth Algorithm.
    FP-Growth compresses transactions into an FP-Tree, avoiding expensive candidate generation.

    Args:
        basket_matrix (pd.DataFrame): One-hot encoded transaction matrix.
        min_support (float): Minimum support threshold fraction [0.0, 1.0]. Default is 0.02.
        min_threshold (float): Minimum threshold for rule extraction metric. Default is 0.2.
        metric (str): Metric to filter rules ('confidence', 'lift', 'support'). Default is 'confidence'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - frequent_itemsets: DataFrame of itemsets meeting min_support.
            - rules: Formatted DataFrame of association rules.
    """
    if basket_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Step 1: Mine frequent itemsets via FP-Growth
    frequent_itemsets = fpgrowth(
        basket_matrix,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["support", "itemsets"]), pd.DataFrame()

    # Step 2: Generate association rules
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold
    )

    if not rules.empty:
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules["rule"] = rules["antecedents_str"] + " ➔ " + rules["consequents_str"]
        
        for col in ["support", "confidence", "lift", "leverage", "conviction"]:
            if col in rules.columns:
                rules[col] = rules[col].round(4)

    return frequent_itemsets, rules


def benchmark_algorithms(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.02
) -> Dict[str, Any]:
    """
    Executes timing benchmarking comparing Apriori vs FP-Growth performance
    for a given basket matrix and minimum support threshold.

    Args:
        basket_matrix (pd.DataFrame): Binary one-hot encoded matrix.
        min_support (float): Minimum support threshold.

    Returns:
        Dict[str, Any]: Benchmarking summary including execution time in seconds,
                        frequent itemset counts, and speedup factor.
    """
    if basket_matrix.empty:
        return {
            "apriori_time_sec": 0.0,
            "fpgrowth_time_sec": 0.0,
            "apriori_itemset_count": 0,
            "fpgrowth_itemset_count": 0,
            "speedup_factor": 1.0
        }

    # Benchmark Apriori
    t0_apriori = time.perf_counter()
    ap_itemsets = apriori(basket_matrix, min_support=min_support, use_colnames=True)
    t1_apriori = time.perf_counter()
    apriori_time = round(t1_apriori - t0_apriori, 5)

    # Benchmark FP-Growth
    t0_fp = time.perf_counter()
    fp_itemsets = fpgrowth(basket_matrix, min_support=min_support, use_colnames=True)
    t1_fp = time.perf_counter()
    fpgrowth_time = round(t1_fp - t0_fp, 5)

    speedup = round(apriori_time / max(fpgrowth_time, 1e-6), 2)

    return {
        "apriori_time_sec": apriori_time,
        "fpgrowth_time_sec": fpgrowth_time,
        "apriori_itemset_count": len(ap_itemsets),
        "fpgrowth_itemset_count": len(fp_itemsets),
        "speedup_factor": speedup
    }
