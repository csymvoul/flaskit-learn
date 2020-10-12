import io
from flask import Flask, Response
from flaskit.flaskit_learn import Clustering
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test_random_key'

c_kmeans = Clustering()

def load_dataset():
    res = c_kmeans.load_data()
    print(res)

@app.route("/")
def index():
    return 'Welcome to Flaskit Learn!'

@app.route('/clustering/kmeans/fit')
def fit_kmeans():
    load_dataset()
    fig_kmeans = c_kmeans.fit_kmeans()
    output = io.BytesIO()
    FigureCanvas(fig_kmeans).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/clustering/kmeans/predict')
def predict_kmeans():
    return 'Predict K-means!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)