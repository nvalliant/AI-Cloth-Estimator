import os
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import cv2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

model = tf.keras.models.load_model('T-ShirtClassifier.keras')

@app.route('/')
def home():
    return render_template('home.html')

# 2. Define the "endpoint" your HTML will talk to
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get('image')
        size = request.form.get('size')
        width = request.form.get('width')
        qty = int(request.form.get('qty', 1))

        if not file:
            return jsonify({'error': 'Gambar tidak ditemukan'}), 400

        # 2. Proses Gambar (Pre-processing)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        test_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        test_img = cv2.resize(test_img, (224, 224))
        test_img = test_img.astype(np.float32)
        input_data = np.expand_dims(test_img, axis=0)
        print(f"Shape: {input_data.shape}")
        print(f"Min Val: {input_data.min()}, Max Val: {input_data.max()}")

        result = model.predict(input_data)
        r = np.argmax(result)
        
        class_names = {0: 'Boxy Fit', 1: 'Oversize Fit', 2: 'Regular Fit', 3: 'Slim Fit'}
        model_kaos = class_names[r]

        df = pd.read_csv('tshirt_size_dataset_400.csv', encoding = 'latin-1')

        df.columns = df.columns.str.strip() 
        target_col = 'Ukuran Kain'

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        X = pd.get_dummies(X, columns=['Type', 'Size'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        model_columns = X_train.columns

        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE: {mse:.2f}")

        def predict_fabric_needs(type_kaos, size, other_features):
            input_data = {
                'Type': [type_kaos],
                'Size': [size]
            }
            input_data.update(other_features)
            
            input_df = pd.DataFrame(input_data)
            
            input_df = pd.get_dummies(input_df, columns=['Type', 'Size'])
            
            input_df = input_df.reindex(columns=model_columns, fill_value=0)
            
            prediction = rf_model.predict(input_df)
            return prediction[0]
        
        kain_per_pcs = predict_fabric_needs(
            type_kaos=model_kaos, 
            size=size, 
            other_features={
                'Lebar Kain': width,
            }
        )

        return jsonify({
            "prediction": model_kaos,
            'total_meter': kain_per_pcs*qty,
            'per_pcs': kain_per_pcs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
