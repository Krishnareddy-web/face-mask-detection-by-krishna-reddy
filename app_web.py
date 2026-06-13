import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import cv2
from detector import detect_and_predict_mask
import base64
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the image
        image = cv2.imread(filepath)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'})

        # Detect masks
        boxes, preds = detect_and_predict_mask(image)
        
        # Draw bounding boxes
        for (box, pred) in zip(boxes, preds):
            x, y, w, h = box
            mask_prob, no_mask_prob = pred
            
            if mask_prob > 0.99:
                color = (0, 255, 0)
                label = "Mask"
            elif mask_prob < 0.5:
                color = (0, 0, 255)
                label = "No Mask"
            else:
                color = (0, 255, 255)
                label = "Uncertain"
                
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label}: {max(mask_prob, no_mask_prob)*100:.2f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Save the processed image with a 'result_' prefix
        result_filename = f"result_{filename}"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, image)
        
        # Return the URL for the frontend to display
        result_url = url_for('static', filename=f'uploads/{result_filename}')
        return jsonify({'success': True, 'result_url': result_url})

    return jsonify({'error': 'Invalid file type'})

@app.route('/predict_live', methods=['POST'])
def predict_live():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'})
    
    # Extract base64 image data
    image_data = data['image'].split(',')[1]
    
    # Decode image
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'})
        
    # Detect masks
    boxes, preds = detect_and_predict_mask(image)
    
    results = []
    for (box, pred) in zip(boxes, preds):
        x, y, w, h = box
        mask_prob, no_mask_prob = pred
        
        # Convert types to standard Python int/float for JSON serialization
        x, y, w, h = int(x), int(y), int(w), int(h)
        mask_prob, no_mask_prob = float(mask_prob), float(no_mask_prob)
        
        if mask_prob > 0.99:
            color = "#00FF00"
            label = "Mask"
        elif mask_prob < 0.5:
            color = "#FF0000"
            label = "No Mask"
        else:
            color = "#FFFF00"
            label = "Uncertain"
            
        results.append({
            "box": [x, y, w, h],
            "label": label,
            "probability": max(mask_prob, no_mask_prob) * 100,
            "color": color
        })
        
    return jsonify({'success': True, 'predictions': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
