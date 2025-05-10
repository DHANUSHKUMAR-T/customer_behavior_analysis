import pandas as pd

def calculate_clv(df):
    if 'Frequency of Purchases' not in df.columns:
        df['Frequency of Purchases'] = 1

    avg_transaction = df.groupby('Customer ID')['Purchase Amount (USD)'].mean()
    purchase_freq = df.groupby('Customer ID')['Frequency of Purchases'].mean()
    lifespan = 12  # months

    clv = avg_transaction * purchase_freq * lifespan
    clv_df = clv.reset_index()
    clv_df.columns = ['Customer ID', 'CLV']
    clv_df['CLV Segment'] = pd.qcut(clv_df['CLV'], q=3, labels=['Low', 'Medium', 'High'])
    return clv_df
