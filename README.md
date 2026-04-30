# 🛍️ AI Customer Segmentation & Marketing Strategy Dashboard

## 📌 Project Overview
This project is an end-to-end customer segmentation system using **SQL, RFM Analysis, K-Means Clustering, PCA visualization, and Streamlit deployment**.

The main goal is not only to create clusters, but to convert those clusters into useful business segments such as **High Value Customers, Regular Customers, Inactive Customers, and Low Spend Customers**.

---

## 🧠 Business Problem
Businesses often have many customers but do not clearly know:

- Which customers are most valuable
- Which customers are becoming inactive
- Which customers need discounts or re-targeting
- Which customers should receive loyalty offers

This project solves that problem using data-driven customer segmentation.

---

## ✅ Key Features
- Data cleaning and preprocessing
- SQL-based business analysis
- RFM feature engineering
- K-Means customer clustering
- Silhouette score for cluster quality
- PCA-based cluster visualization
- Customer segment distribution chart
- Business recommendations for each segment
- Downloadable segmented customer CSV
- Downloadable business report
- Streamlit dashboard deployment

---

## ⚙️ Tech Stack
- Python
- Pandas
- NumPy
- SQLite / SQL
- Scikit-learn
- Matplotlib
- Streamlit

---

## 🔄 Project Workflow
1. Load online retail transaction data
2. Clean missing values and duplicates
3. Remove invalid transactions
4. Create total amount feature
5. Perform SQL business analysis
6. Create RFM features:
   - Recency
   - Frequency
   - Monetary
7. Apply log transformation and scaling
8. Train K-Means clustering model
9. Evaluate clusters using silhouette score
10. Visualize clusters using PCA
11. Assign business names to customer segments
12. Generate marketing recommendations
13. Deploy using Streamlit

---

## 🗄️ SQL Analysis Performed
- Total revenue analysis
- Top 10 customers by spending
- Top 10 products by revenue
- Monthly revenue trend

---

## 📊 RFM Features
| Feature | Meaning |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of purchases/orders |
| Monetary | Total customer spending |

---

## 🤖 Machine Learning Model
The project uses **K-Means Clustering**, an unsupervised machine learning algorithm, to group customers based on purchasing behavior.

---

## 📈 Evaluation
The project uses **Silhouette Score** to evaluate how well customer clusters are separated.

---

## 💼 Business Impact
This project helps a business:

- Retain high-value customers using loyalty rewards
- Win back inactive customers using reactivation campaigns
- Increase revenue from low-spend customers using bundle offers
- Improve targeted marketing using customer behavior
- Make better customer relationship decisions

---

## 🚀 Streamlit Dashboard Features
- Upload CSV or use included sample data
- View raw and cleaned data
- See important business KPIs
- Run SQL analysis
- Generate RFM customer table
- Select number of clusters
- View silhouette score
- Visualize customer segments
- Read business recommendations
- Download segmented customers and report

---

## ▶️ How to Run

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🧾 Dataset
The project uses online retail transaction data with columns such as:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

---

## 🎯 Interview Explanation
In this project, I used RFM analysis and K-Means clustering to segment customers based on purchasing behavior. I first cleaned the transaction data, performed SQL business analysis, created Recency, Frequency and Monetary features, scaled the data, and applied K-Means clustering.

The important part of this project is that I converted technical clusters into business-friendly customer segments and added marketing recommendations for each segment. This makes the project useful for real-world business decision-making.

---

## 📌 Future Improvements
- Add customer lifetime value prediction
- Add churn prediction
- Add automated email campaign suggestions
- Deploy on cloud platform
