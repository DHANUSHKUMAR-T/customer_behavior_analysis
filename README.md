# 🧠 Customer Behavior Analysis Dashboard

This Flask-based dashboard enables businesses to analyze customer behavior using advanced data science techniques. It supports data upload, segmentation, lifetime value analysis, and visual trend reporting.

---

## 📌 Features

- ✅ **K-Means Clustering** – Group customers based on purchasing behavior  
- 📈 **Customer Lifetime Value (CLV)** – Quantify long-term customer worth  
- 🔍 **High-Value Customer Detection** – Identify loyal or big-spending users  
- 📊 **Category & Time Trends** – Understand sales dynamics and patterns  
- 🗃️ **Interactive Dashboard** – Built with Flask + Plotly for quick insights  
- 🧾 **Report Exporting** – Download CSV summaries and charts  

---

## 📂 Sample CSV Format

Make sure your CSV contains the following columns:

| Column              | Description                                          |
|---------------------|------------------------------------------------------|
| `Customer ID`       | Unique identifier for each customer                  |
| `Purchase Amount`   | Transaction amount in USD                            |
| `Frequency`         | Number of transactions/purchases by the customer     |
| `Product Category`  | *(Optional)* Category or type of product purchased   |
| `Date`              | *(Optional but preferred)* Date of transaction       |

> ⚠️ Ensure proper formatting: no missing values and correct data types.

---

## 🚀 How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/customer-dashboard.git
   cd customer-dashboard

2. Install dependencies:
pip install -r requirements.txt

3. Run the Flask app:
python app.py


📊 Output Highlights
Customer Segments based on similarity in purchase behavior

Top Customers ranked by estimated lifetime value

Purchase Trends by category and time

Visual Charts: Scatter plots, bar charts, and CLV graphs

🧠 Tech Stack
Backend: Python, Flask

Data Analysis: Pandas, Scikit-learn

Visualization: Plotly, Matplotlib

Frontend: HTML, CSS (Bootstrap)

🔒 Data Privacy
No customer data is stored permanently. For production deployment, ensure encryption, secure sessions, and access control mechanisms.