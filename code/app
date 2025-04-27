from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import cv2
import torch
from werkzeug.utils import secure_filename

from color_detector import detect_watch_color  # Make sure this function exists

# Configuration
UPLOAD_FOLDER = 'static/uploads'
SAVED_ITEMS_FILE = 'saved_items.txt'

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv5 model once (at app start)
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='C:/lostandfound/yolov5/runs/train/exp/weights/best.pt',  # Change if needed
    force_reload=True
)

# Save detected item info to text file
def save_item(item, color, filename):
    try:
        with open(SAVED_ITEMS_FILE, 'a') as f:
            safe_item = item.strip().lower()
            safe_color = color.strip().lower()
            safe_filename = filename.strip()
            f.write(f"{safe_item},{safe_color},{safe_filename}\n")

    except Exception as e:
        print(f"Error saving item: {e}")

# Search for matching item
def search_items(item_name, color):
    try:
        if os.path.exists(SAVED_ITEMS_FILE):
            with open(SAVED_ITEMS_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 3:
                        continue
                    saved_item, saved_color, saved_filename = parts
                    if saved_item.strip().lower() == item_name and saved_color.strip().lower() == color:
                        return saved_filename
    except Exception as e:
        print(f"Error searching items: {e}")
    return None

@app.route('/')
def index():
    return render_template('index.html', result_message='')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files.get('image')
    if not image:
        return jsonify({"success": False, "message": "No image uploaded."})

    # Save uploaded image
    raw_name = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + image.filename
    filename = secure_filename(raw_name).replace(" ", "_")  # Clean filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
 # Run YOLOv5 detection
    results = model(filepath)
    boxes = results.pandas().xyxy[0]

    if boxes.empty:
        return jsonify({"success": False, "message": "No watch detected."})

    # Take the first detected object
    row = boxes.iloc[0]
    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

    # Crop detected watch region
    img_cv = cv2.imread(filepath)
    cropped_img = img_cv[ymin:ymax, xmin:xmax]

    # Save cropped image (optional)
    cropped_path = os.path.join('static', 'cropped.jpg')
    cv2.imwrite(cropped_path, cropped_img)

    # Detect color
    box = (xmin, ymin, xmax, ymax)
    detected_color = detect_watch_color(filepath, box)
    detected_item = "watch"

    # Save item
    save_item(detected_item, detected_color, filename)

    return jsonify({
        "success": True,
        "message": f"Upload successful! Detected a {detected_color} {detected_item}.",
        "color": detected_color,
        "image_url": f"/static/uploads/{filename}"
    })

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        item_name = data.get('item_name', '').strip().lower()
        color = data.get('color', '').strip().lower()

        if not item_name or not color:
            return jsonify({"result": "Please provide both item name and color.", "image_url": ""})

        matched_filename = search_items(item_name, color)
        if matched_filename:
            return jsonify({
                "result": f"Found a matching {color} {item_name}!",
                "image_url": f"/static/uploads/{matched_filename}"
            })
        else:
            return jsonify({
                "result": f"No {color} {item_name} found.",
                "image_url": ""
            })

    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"result": "An error occurred while searching.", "image_url": ""})

if _name_ == '_main_':
    app.run(debug=True)
