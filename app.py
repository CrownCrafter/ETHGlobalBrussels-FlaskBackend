from flask import Flask
from flask import jsonify
from sklearn.cluster import KMeans
from flask_cors import CORS
from urllib.request import urlretrieve, urlopen
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import send_file
from sklearn.decomposition import PCA
from flask import request
from flask_cors import cross_origin
app = Flask(__name__)
CORS(app, resources=r'/api/*')

app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/api/")
def hello_world():
    return jsonify("test")

@app.route("/api/basic_data")
async def basic_data():
    with urlopen('https://gateway.lighthouse.storage/ipfs/' + request.args.get('cid')) as f: # Download File
        df = pd.read_csv(f)

        # Summary Statistics
        summary = df.describe().to_dict()

        # Missing Values
        missing_values = df.isnull().sum().to_dict()

        # Unique Values
        unique_values = df.nunique().to_dict()

        # Histograms
        histograms = {}
        for column in df.select_dtypes(include=['number']).columns:
            plt.figure()
            df[column].hist()
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            histograms[column] = base64.b64encode(img.getvalue()).decode()
            plt.close()

        # # Correlation Matrix
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        correlation_matrix = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({
            'summary': summary,
            'missing_values': missing_values,
            'unique_values': unique_values,
            # 'histograms': histograms,
            # 'correlation_matrix': correlation_matrix
        })
@app.route('/api/basic_data_corr')
async def basic_data_corr():
    with urlopen('https://gateway.lighthouse.storage/ipfs/' + request.args.get('cid')) as f: # Download File

        df = pd.read_csv(f)
         # # Correlation Matrix
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        correlation_matrix = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        return jsonify({"image":correlation_matrix})


@app.route('/api/pca')
async def pca():

    """
    Perform PCA on the provided dataset to determine feature importance.

    Parameters:
    data (list of dict): List of dictionaries containing the dataset.
    n_components (int): Number of principal components to compute. Default is None, which means all components are computed.

    Returns:
    dict: A dictionary containing the explained variance ratio and feature importance.
    """

    with urlopen('https://gateway.lighthouse.storage/ipfs/' + request.args.get('cid')) as f: # Download File
        n_components = 3
        df = pd.read_csv(f)
        numeric_features = df.select_dtypes(include='number').columns.tolist()
        selected_features = df[numeric_features]

        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(selected_features)

        explained_variance_ratio = pca.explained_variance_ratio_
        feature_importance = pd.DataFrame(pca.components_, columns=numeric_features).abs().mean().sort_values(ascending=False)

        pca_result = {
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'feature_importance': feature_importance.to_dict()
        }
        # Create DataFrame for explained variance ratio
        explained_variance_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(pca_result['explained_variance_ratio']))],
            'Explained Variance Ratio': pca_result['explained_variance_ratio']
        })
        # Create DataFrame for feature importance
        feature_importance_df = pd.DataFrame(list(pca_result['feature_importance'].items()), columns=['Feature', 'Importance'])

        # Display explained variance ratio as HTML table
        # return (explained_variance_df.to_html(classes='table table-striped', index=False), feature_importance_df.to_html(classes='table table-striped', index=False))
        return jsonify(explained_variance_df.to_dict())


@app.route("/api/2dkmeans")
def knn(url:str, feature_x:str, feature_y:str):

    with urlopen(url,) as f: # Download File
        n_clusters = 3
        df = pd.read_csv(f)
        selected_features = df[[feature_x, feature_y]]

        kmeans = KMeans(n_clusters=n_clusters)
        df['cluster'] = kmeans.fit_predict(selected_features)

        plt.figure(figsize=(8, 6))
        plt.scatter(df[feature_x], df[feature_y], c=df['cluster'])
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f'KMeans Clustering with {n_clusters} Clusters')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        kmeans_plot = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        return jsonify([{
            'kmeans_plot': kmeans_plot
        }])

@app.route("/api/vis")
def vis():
    return ""

if __name__ == '__main__':
    app.run(debug=True)


