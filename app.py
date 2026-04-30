import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="AI Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1E3A8A;
    font-size: 42px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    color: #64748B;
    font-size: 18px;
}
.highlight-card {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #BFDBFE;
    margin-bottom: 12px;
}
.stButton>button {
    background: linear-gradient(90deg, #2563EB, #4F46E5);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1D4ED8, #4338CA);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛍️ AI Customer Segmentation & Marketing Strategy Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>SQL + RFM Analysis + K-Means Clustering + Business Recommendations</div>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("📌 Project Highlights")
    st.markdown("""
    - Customer segmentation using **RFM Analysis**
    - **SQL business analysis** on retail transactions
    - **K-Means clustering** for customer grouping
    - **Silhouette score** for cluster quality
    - **PCA visualization** for cluster separation
    - Segment-wise **marketing strategy**
    - Downloadable segmented customer report
    """)
    st.info("Tip: Upload Online Retail CSV or use the sample data included in the project.")

uploaded_file = st.file_uploader("📂 Upload Online Retail CSV file", type=["csv"])

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv("customer_data_cleaned.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def clean_data(dataframe):
    df = dataframe.copy()
    df = df.dropna(subset=["CustomerID"])
    df = df.drop_duplicates()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df

@st.cache_data
def create_rfm(dataframe):
    snapshot_date = dataframe["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = dataframe.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    rfm["Recency_log"] = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    return rfm

def assign_segment_names(summary):
    segment_names = {}
    monetary_rank = summary["Monetary"].rank(ascending=False, method="dense")
    frequency_rank = summary["Frequency"].rank(ascending=False, method="dense")
    recency_rank = summary["Recency"].rank(ascending=True, method="dense")

    for cluster in summary.index:
        if monetary_rank.loc[cluster] == 1 and frequency_rank.loc[cluster] <= 2:
            segment_names[cluster] = "High Value Customers 💰"
        elif recency_rank.loc[cluster] == len(summary):
            segment_names[cluster] = "Inactive Customers ⚠️"
        elif monetary_rank.loc[cluster] >= len(summary) - 1:
            segment_names[cluster] = "Low Spend Customers 🧊"
        else:
            segment_names[cluster] = "Regular Customers 🙂"
    return segment_names

def recommendation_for_segment(segment):
    if "High Value" in segment:
        return "Give loyalty rewards, premium offers, early access, and VIP support to retain these customers."
    if "Inactive" in segment:
        return "Run win-back campaigns, limited-time discounts, and personalized reminders."
    if "Low Spend" in segment:
        return "Use bundle offers, small discounts, and cross-sell campaigns to increase order value."
    return "Use engagement campaigns, product recommendations, and seasonal offers to increase frequency."

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = load_sample_data()

if raw_df is None:
    st.warning("Please upload your Online Retail CSV file to start.")
    st.stop()

required_columns = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
missing_columns = required_columns - set(raw_df.columns)
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

st.success("Data loaded successfully!" if uploaded_file else "Sample project data loaded successfully!")

with st.expander("🔍 View Raw Data Preview"):
    st.dataframe(raw_df.head(20), use_container_width=True)

df = clean_data(raw_df)

st.subheader("🧹 Data Cleaning Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clean Transactions", f"{len(df):,}")
col2.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
col3.metric("Total Revenue", f"₹{df['TotalAmount'].sum():,.0f}")
col4.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")

st.markdown("---")

st.subheader("🗄️ SQL Business Analysis")
conn = sqlite3.connect(":memory:")
df.to_sql("retail_data", conn, if_exists="replace", index=False)

top_customers = pd.read_sql("""
SELECT CustomerID, ROUND(SUM(TotalAmount), 2) AS total_spent
FROM retail_data
GROUP BY CustomerID
ORDER BY total_spent DESC
LIMIT 10;
""", conn)

top_products = pd.read_sql("""
SELECT Description, ROUND(SUM(TotalAmount), 2) AS revenue
FROM retail_data
GROUP BY Description
ORDER BY revenue DESC
LIMIT 10;
""", conn)

monthly_revenue = pd.read_sql("""
SELECT strftime('%Y-%m', InvoiceDate) AS month,
       ROUND(SUM(TotalAmount), 2) AS monthly_revenue
FROM retail_data
GROUP BY month
ORDER BY month;
""", conn)

col5, col6 = st.columns(2)
with col5:
    st.write("🏆 Top 10 Customers by Spending")
    st.dataframe(top_customers, use_container_width=True)
with col6:
    st.write("📦 Top 10 Products by Revenue")
    st.dataframe(top_products, use_container_width=True)

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(monthly_revenue["month"], monthly_revenue["monthly_revenue"], marker="o")
ax1.set_title("Monthly Revenue Trend")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue")
plt.xticks(rotation=45)
st.pyplot(fig1)

st.markdown("---")

st.subheader("📊 RFM Feature Engineering")
rfm = create_rfm(df)
st.dataframe(rfm[["CustomerID", "Recency", "Frequency", "Monetary"]].head(15), use_container_width=True)

features = rfm[["Recency_log", "Frequency_log", "Monetary_log"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

st.subheader("⚙️ Cluster Selection")
k = st.slider("Choose number of clusters", min_value=2, max_value=8, value=4)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(scaled_features)
sil_score = silhouette_score(scaled_features, rfm["Cluster"])

cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
segment_names = assign_segment_names(cluster_summary)
rfm["SegmentName"] = rfm["Cluster"].map(segment_names)
rfm["MarketingRecommendation"] = rfm["SegmentName"].apply(recommendation_for_segment)

st.success("Customer segmentation completed successfully!")
st.metric("Silhouette Score", round(sil_score, 3))
st.caption("Higher silhouette score generally means better cluster separation.")

st.subheader("📌 Cluster Summary with Business Meaning")
summary_for_display = cluster_summary.copy()
summary_for_display["Segment Name"] = summary_for_display.index.map(segment_names)
summary_for_display["Business Recommendation"] = summary_for_display["Segment Name"].apply(recommendation_for_segment)
st.dataframe(summary_for_display, use_container_width=True)

st.markdown("---")

col7, col8 = st.columns(2)
with col7:
    st.subheader("📈 Customer Segment Distribution")
    segment_counts = rfm["SegmentName"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(segment_counts.index, segment_counts.values)
    ax2.set_title("Customer Segment Distribution")
    ax2.set_xlabel("Segment")
    ax2.set_ylabel("Number of Customers")
    plt.xticks(rotation=35, ha="right")
    st.pyplot(fig2)

with col8:
    st.subheader("📊 PCA Cluster Visualization")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    rfm["PCA1"] = pca_features[:, 0]
    rfm["PCA2"] = pca_features[:, 1]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    scatter = ax3.scatter(rfm["PCA1"], rfm["PCA2"], c=rfm["Cluster"], alpha=0.7)
    ax3.set_xlabel("PCA Component 1")
    ax3.set_ylabel("PCA Component 2")
    ax3.set_title("Customer Segments using PCA")
    fig3.colorbar(scatter, ax=ax3, label="Cluster")
    st.pyplot(fig3)

st.subheader("💼 Segment-wise Marketing Strategy")
for segment in rfm["SegmentName"].unique():
    count = (rfm["SegmentName"] == segment).sum()
    st.markdown(f"""
    <div class='highlight-card'>
    <h4>{segment}</h4>
    <p><b>Total Customers:</b> {count}</p>
    <p><b>Strategy:</b> {recommendation_for_segment(segment)}</p>
    </div>
    """, unsafe_allow_html=True)

st.subheader("📄 Final Segmented Customers")
st.dataframe(rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "SegmentName", "MarketingRecommendation"]].head(50), use_container_width=True)

csv = rfm.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Segmented Customers CSV",
    data=csv,
    file_name="customer_segments_with_strategy.csv",
    mime="text/csv"
)

report_text = f"""
Customer Segmentation Business Report

Total Customers: {df['CustomerID'].nunique():,}
Total Revenue: ₹{df['TotalAmount'].sum():,.2f}
Total Orders: {df['InvoiceNo'].nunique():,}
Selected Clusters: {k}
Silhouette Score: {sil_score:.3f}

Segment Summary:
{summary_for_display.to_string()}

Conclusion:
This dashboard groups customers based on Recency, Frequency and Monetary value. The final segments help the business identify high-value customers, inactive customers, regular customers and low-spend customers. These insights can be used for loyalty rewards, retention campaigns, win-back offers and targeted marketing.
"""

st.download_button(
    label="📄 Download Business Report",
    data=report_text,
    file_name="customer_segmentation_business_report.txt",
    mime="text/plain"
)
