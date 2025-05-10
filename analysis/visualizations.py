import plotly.express as px
import plotly
import json

def plot_cluster_distribution(df):
    return px.histogram(df, x='Cluster', title='Customer Cluster Distribution')

def plot_high_value_customers(df):
    high = df[df['Cluster'] == df['Cluster'].max()]
    return px.scatter(high, x='Purchase Amount (USD)', y='Frequency of Purchases',
                      title='High Value Customers')

def plot_category_trends(df):
    if 'Category' not in df.columns:
        return px.scatter(title='No Category Data Available')
    return px.histogram(df, x='Category', color='Cluster', barmode='group',
                        title='Category Trends by Cluster')


def plot_clv_segments(clv_df):
    return px.pie(clv_df, names='CLV Segment', title='Customer Lifetime Value Segments')
