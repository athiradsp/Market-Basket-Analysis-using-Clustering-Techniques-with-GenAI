"""
Data Processing & Feature Engineering Module for Market Basket Analysis (CSM 355).

This module handles:
1. Synthetic data generation with realistic retail co-occurrence patterns.
2. Data cleaning (null removal, date parsing, return/cancellation filtering).
3. Feature engineering for Customer Segmentation (RFM Metrics & Standard Scaling).
4. Transaction matrix transformation for Association Rule Mining (One-Hot Encoded Basket Matrix).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(num_records: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic transactional dataset mimicking the UCI Online Retail Dataset.
    Includes intentional data anomalies (null CustomerIDs, returns starting with 'C', negative quantities)
    to test the data cleaning pipeline.

    Args:
        num_records (int): Total number of transaction rows to generate. Default is 1500.
        random_seed (int): Seed for reproducible pseudorandom generation.

    Returns:
        pd.DataFrame: Raw synthetic transactional DataFrame.
    """
    np.random.seed(random_seed)
    
    # Co-purchased item bundles (baskets) to ensure strong association rules
    item_bundles = [
        [("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
         ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
         ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69)],
        
        [("20725", "LUNCH BAG RED RETROSPOT", 1.65),
         ("20727", "LUNCH BAG BLACK SKULL", 1.65),
         ("20728", "LUNCH BAG CARS BLUE", 1.65)],
        
        [("22383", "LUNCH BAG SUKI DESIGN", 1.65),
         ("22384", "LUNCH BAG PINK POLKADOT", 1.65)],
        
        [("22960", "JAM MAKING SET WITH JARS", 4.25),
         ("22961", "JAM MAKING SET PRINTED", 4.25),
         ("23310", "SET 3 RETROSPOT TEA,COFFEE,SUGAR", 4.95)],
        
        [("84991", "60 TEATIME MATCHES RD", 0.42),
         ("84992", "72 SWEETHEART FAIRY CAKE CASES", 0.55),
         ("21212", "PACK OF 60 PINK PAISLEY CAKE CASES", 0.55)],
        
        [("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
         ("22722", "SET OF 6 SPICE TINS PANTRY DESIGN", 3.95)],
        
        [("21080", "SET/20 RED RETROSPOT PAPER NAPKINS", 0.85),
         ("21086", "SET/60 I LOVE LONDON CAKE CASES", 0.55),
         ("21094", "SET/6 RED SPOTTY PAPER PLATES", 0.85)]
    ]

    customers = [f"1{i:04d}" for i in range(1, 151)]  # 150 unique customers
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    
    records = []
    invoice_counter = 500000

    while len(records) < num_records:
        invoice_counter += 1
        invoice_no = str(invoice_counter)
        cust_id = np.random.choice(customers) if np.random.rand() > 0.08 else None  # 8% missing customers
        inv_date = np.random.choice(dates)
        
        # Pick a random product bundle or random individual items
        bundle = item_bundles[np.random.randint(0, len(item_bundles))]
        # Select 1 to len(bundle) items from bundle to simulate basket variation
        basket_size = np.random.randint(1, len(bundle) + 1)
        selected_items = [bundle[i] for i in np.random.choice(len(bundle), size=basket_size, replace=False)]
        
        for stock_code, desc, unit_price in selected_items:
            # Introduce intentional return/cancellation logic (~3% of rows)
            if np.random.rand() < 0.03:
                inv_no_curr = f"C{invoice_no}"
                qty = -int(np.random.randint(1, 5))
            else:
                inv_no_curr = invoice_no
                qty = int(np.random.randint(1, 12))
                
            records.append({
                "InvoiceNo": inv_no_curr,
                "StockCode": stock_code,
                "Description": desc,
                "Quantity": qty,
                "InvoiceDate": inv_date,
                "UnitPrice": unit_price,
                "CustomerID": cust_id
            })

    df = pd.DataFrame(records[:num_records])
    
    # Introduce negative prices for testing defensive cleaning (~1% of rows)
    negative_price_mask = np.random.rand(len(df)) < 0.01
    df.loc[negative_price_mask, "UnitPrice"] = -1.0
    
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw retail transactional data by enforcing strict cleaning rules:
    - Removing missing CustomerIDs and Descriptions.
    - Converting InvoiceDate to datetime objects.
    - Filtering out canceled orders (InvoiceNo starting with 'C').
    - Removing non-positive Quantities and UnitPrices.
    - Computing total purchase value per item (TotalPrice = Quantity * UnitPrice).

    Args:
        df (pd.DataFrame): Raw transactional DataFrame.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Cleaned DataFrame ready for RFM & Market Basket processing.
            - Audit summary dictionary containing statistics on dropped records.
    """
    initial_rows = len(df)
    
    # Defensive copy
    cleaned = df.copy()

    # Step 0: Column name normalization & auto-mapping
    col_map = {col: str(col).strip() for col in cleaned.columns}
    cleaned.rename(columns=col_map, inplace=True)

    standard_map = {}
    for col in cleaned.columns:
        norm = str(col).lower().replace("_", "").replace(" ", "").replace("-", "")
        if norm in ['invoiceno', 'invoice', 'invoicenumber', 'invno']:
            standard_map[col] = 'InvoiceNo'
        elif norm in ['stockcode', 'itemcode', 'productcode', 'sku', 'code']:
            standard_map[col] = 'StockCode'
        elif norm in ['description', 'desc', 'itemdescription', 'productname', 'item', 'product']:
            standard_map[col] = 'Description'
        elif norm in ['quantity', 'qty', 'units', 'count', 'amount']:
            standard_map[col] = 'Quantity'
        elif norm in ['invoicedate', 'date', 'timestamp', 'transactiondate', 'time']:
            standard_map[col] = 'InvoiceDate'
        elif norm in ['unitprice', 'price', 'rate', 'itemprice', 'cost']:
            standard_map[col] = 'UnitPrice'
        elif norm in ['customerid', 'customer', 'userid', 'clientid', 'custid', 'user']:
            standard_map[col] = 'CustomerID'
    
    cleaned.rename(columns=standard_map, inplace=True)

    required_cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [c for c in required_cols if c not in cleaned.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID."
        )

    # Step 1: Handle missing values
    missing_cust = cleaned["CustomerID"].isna().sum()
    missing_desc = cleaned["Description"].isna().sum()
    cleaned = cleaned.dropna(subset=["CustomerID", "Description"])

    # Step 2: Ensure correct data types
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(str).str.replace(".0", "", regex=False)
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")

    # Step 3: Remove return transactions starting with 'C'
    returns_count = cleaned["InvoiceNo"].str.startswith("C", na=False).sum()
    cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C", na=False)]

    # Step 4: Remove non-positive quantities and prices
    invalid_qty_price = len(cleaned[(cleaned["Quantity"] <= 0) | (cleaned["UnitPrice"] <= 0)])
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]

    # Step 5: Feature engineering - Line Total
    cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    # Summary audit report
    audit_stats = {
        "initial_rows": initial_rows,
        "cleaned_rows": len(cleaned),
        "missing_customer_ids": int(missing_cust),
        "missing_descriptions": int(missing_desc),
        "canceled_invoices": int(returns_count),
        "invalid_qty_or_price": int(invalid_qty_price),
        "retention_rate_pct": round((len(cleaned) / max(initial_rows, 1)) * 100, 2)
    }

    return cleaned, audit_stats


def compute_rfm(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Recency, Frequency, and Monetary (RFM) metrics per CustomerID.

    Mathematical Metrics:
    - Recency (R): Days elapsed between customer's last purchase and snapshot date.
                   R = (max(InvoiceDate) + 1 day) - max(Customer InvoiceDate)
    - Frequency (F): Total count of distinct invoices placed by customer.
                     F = count(distinct InvoiceNo)
    - Monetary (M): Total net revenue generated by customer across all transactions.
                    M = sum(Quantity * UnitPrice)

    Features are log-transformed (to reduce skewness) and normalized using StandardScaler.

    Args:
        df (pd.DataFrame): Cleaned transactional DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - rfm_raw: Raw RFM DataFrame indexed by CustomerID.
            - rfm_scaled: Standardized RFM DataFrame suitable for distance-based clustering.
    """
    if df.empty:
        empty_rfm = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])
        return empty_rfm, empty_rfm

    # Set snapshot reference date as max invoice date + 1 day
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm_raw = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalPrice": "Monetary"
    })

    # Defensive filtering for non-positive monetary values
    rfm_raw = rfm_raw[rfm_raw["Monetary"] > 0].copy()

    # Apply Log Transformation to treat heavy right skewness in financial data
    rfm_log = np.log1p(rfm_raw)

    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    rfm_scaled_array = scaler.fit_transform(rfm_log)
    
    rfm_scaled = pd.DataFrame(
        rfm_scaled_array,
        index=rfm_raw.index,
        columns=["Recency", "Frequency", "Monetary"]
    )

    return rfm_raw, rfm_scaled


def create_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw retail transactions into a binary one-hot encoded matrix
    (InvoiceNo x Product Description) required for Apriori & FP-Growth basket analysis.

    Args:
        df (pd.DataFrame): Cleaned transactional DataFrame.

    Returns:
        pd.DataFrame: One-hot encoded transactional DataFrame where:
                      - Rows represent unique Invoices.
                      - Columns represent unique Product Descriptions.
                      - Cells are 1 if item was purchased in invoice, else 0.
    """
    if df.empty:
        return pd.DataFrame()

    # Group by InvoiceNo and Description, sum quantity
    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index("InvoiceNo")
    )

    # Convert positive quantities to bool True/False required by mlxtend
    basket_binary = (basket > 0)

    return basket_binary
