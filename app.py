import os
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from analysis.clustering import perform_clustering
from analysis.visualizations import plot_category_trends
from clv import calculate_clv
from analysis.visualizations import (
    plot_cluster_distribution, plot_clv_segments, plot_high_value_customers, plot_category_trends)
import plotly.io as pio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'dataset'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

data_df = None
clustered_df = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global data_df, clustered_df
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            data_df = pd.read_csv(filepath)

            clustered_df = perform_clustering(data_df)

            return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    global clustered_df
    if clustered_df is None:
        return redirect(url_for('index'))

    fig1 = plot_cluster_distribution(clustered_df)
    fig2 = plot_high_value_customers(clustered_df)
    fig3 = plot_category_trends(clustered_df)
    clv_df = calculate_clv(clustered_df)
    fig4 = plot_clv_segments(clv_df)

    graph1 = pio.to_html(fig1, full_html=False)
    graph2 = pio.to_html(fig2, full_html=False)
    graph3 = pio.to_html(fig3, full_html=False)
    graph4 = pio.to_html(fig4, full_html=False)


    return render_template("dashboard.html", graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4)

if __name__ == '__main__':
    app.run(debug=True)