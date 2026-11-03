from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

df = pd.read_csv('customers.csv')
X = df[['AnnualIncome', 'SpendingScore']]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html', data=df.head(100).to_dict(orient='records'))

@app.route('/algorithm')
def algorithm():
    centers = kmeans.cluster_centers_.tolist()
    return render_template('algorithm.html', centers=centers, count=len(df))

@app.route('/universe')
def universe():
    return render_template('universe.html')

@app.route('/api/data')
def api_data():
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/centers')
def api_centers():
    return jsonify(kmeans.cluster_centers_.tolist())

if __name__ == '__main__':
    app.run(debug=True)