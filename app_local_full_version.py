from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            confidence REAL,
            timestamp TEXT,
            feedback TEXT DEFAULT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

interpreter = tf.lite.Interpreter(model_path='model/pneumonia_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    base_model = model.layers[1]
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array)
        x = model.layers[2](base_output)   # GlobalAveragePooling2D
        x = model.layers[3](x)             # Dense(128)
        x = model.layers[4](x)             # Dropout
        preds = model.layers[5](x)         # Final Dense (sigmoid)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

app = Flask(__name__, template_folder='templates_full')

# Load the trained model once when app starts
model = load_model('model/pneumonia_model.h5')
explanations = {
    "NORMAL": "No visible signs of lung opacity or consolidation were detected. The lung fields appear clear.",
    "PNEUMONIA": "Patterns consistent with lung opacity and consolidation were detected, which are commonly associated with pneumonia."
}

last_result = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    # Save uploaded file temporarily
    filepath = os.path.join('static/uploaded_image.jpg')
    file.save(filepath)
    
    # Preprocess image to match training format
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    # Prediction (always uses TFLite)
    img_array = img_array.astype('float32')
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    if prediction > 0.5:
        label = "PNEUMONIA"
        confidence = prediction * 100
    else:
        label = "NORMAL"
        confidence = (1 - prediction) * 100
    
    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    original_img = cv2.imread(filepath)
    original_img = cv2.resize(original_img, (224, 224))
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    
    gradcam_path = 'static/gradcam_output.jpg'
    cv2.imwrite(gradcam_path, superimposed)
    
    explanation = explanations[label]
    # Save to database
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO predictions (label, confidence, timestamp) VALUES (?, ?, ?)',
        (label, round(float(confidence), 2), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    prediction_id = c.lastrowid
    conn.close()
    
    last_result['label'] = label
    last_result['confidence'] = round(float(confidence), 2)
    last_result['explanation'] = explanation
    last_result['image_path'] = filepath
    last_result['gradcam_path'] = gradcam_path
    
    return jsonify({
        'label': label,
        'confidence': round(float(confidence), 2),
        'image_path': filepath,
        'gradcam_path': gradcam_path,
        'explanation': explanation
    })
from fpdf import FPDF
from flask import send_file

@app.route('/download_report')
def download_report():
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'AI Chest X-Ray Pneumonia Detection Report', ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Result: {last_result['label']}", ln=True)
    pdf.cell(0, 10, f"Confidence: {last_result['confidence']}%", ln=True)
    pdf.ln(5)
    
    pdf.multi_cell(0, 8, f"Explanation: {last_result['explanation']}")
    pdf.ln(5)
    
    pdf.image(last_result['image_path'], w=80)
    pdf.ln(5)
    pdf.cell(0, 10, 'Grad-CAM Heatmap (model focus area):', ln=True)
    pdf.image(last_result['gradcam_path'], w=80)
    
    report_path = 'static/report.pdf'
    pdf.output(report_path)
    
    return send_file(report_path, as_attachment=True)

@app.route('/history')
def history():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 10')
    rows = c.fetchall()
    conn.close()
    
    history_list = []
    for row in rows:
        history_list.append({
            'id': row[0],
            'label': row[1],
            'confidence': row[2],
            'timestamp': row[3],
            'feedback': row[4]
        })
    
    return jsonify(history_list)

if __name__ == '__main__':
    app.run(debug=False)