from flask import Flask, request, jsonify
import Titanic

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/train', methods=['POST'])
def train():
    #input_data = request.files['csvFile']

    import pandas as pd
    train_data = pd.read_csv('train.csv', index_col = 0).dropna(subset=['Age','Embarked'])

    Titanic.train_model(train_data)

    return jsonify({'message': 'Модель успешно обучена!'})


@app.route('/predict', methods=['POST'])
def predict():
    #input_data = request.request.json
    
    result = Titanic.predict()

    return jsonify({'fpr': result['fpr'].tolist(), 'tpr': result['tpr'].tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
