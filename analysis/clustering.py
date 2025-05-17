from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def perform_clustering(df, n_clusters=4):
    features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']

    if 'Frequency of Purchases' in df.columns:
        le = LabelEncoder()
        df['Frequency of Purchases'] = le.fit_transform(df['Frequency of Purchases'].astype(str))

    X = df[features].copy()

    X = X.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    print(df.dtypes)
    print(df['Frequency of Purchases'].unique())


    return df
