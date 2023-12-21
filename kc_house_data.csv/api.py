from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('housing_price_model.pkl')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)

        # Convert the JSON data to a DataFrame
        input_data = pd.DataFrame(data)

        # Make predictions
        predictions = model.predict(input_data)

        # Prepare the response
        response = {'predictions': predictions.tolist()}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
