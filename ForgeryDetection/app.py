from flask import Flask, request, jsonify, url_for
import os
import uuid
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from predict_svm import predict_single_image
from highlight_forgery import highlight_forged_area

app = Flask(__name__)

# ✅ Home route
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Image Forgery Detection API!"})


# ✅ Route to compare two images (original and forged)
@app.route('/compare', methods=['POST'])
def compare_images():
    print("📩 Received request on /compare")

    if 'real' not in request.files or 'edited' not in request.files:
        return jsonify({"error": "Both 'real' and 'edited' images are required"}), 400

    real = request.files['real']
    edited = request.files['edited']

    if real.filename == '' or edited.filename == '':
        return jsonify({"error": "Empty file(s) uploaded"}), 400

    try:
        real_path = 'temp_real.jpg'
        edited_path = 'temp_edited.jpg'
        real.save(real_path)
        edited.save(edited_path)

        # Load images in grayscale
        img1 = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(edited_path, cv2.IMREAD_GRAYSCALE)

        # Resize to match dimensions
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute SSIM
        score, diff = ssim(img1, img2, full=True)
        diff = (diff * 255).astype("uint8")

        # Threshold and find contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = cv2.imread(edited_path)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save output image
        output_filename = f"highlighted_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join('static', output_filename)
        cv2.imwrite(output_path, output)

        output_url = url_for('static', filename=output_filename)
        return jsonify({
            'ssim_score': f"{score:.4f}",
            'highlighted_image_url': request.host_url + output_url
        })

    except Exception as e:
        print("❌ Error during image comparison:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ Route to predict forgery in a single image using SVM
@app.route('/predict', methods=['POST'])
def predict():
    print("📩 Received request on /predict")

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = request.files['image']
    if img.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image_path = 'temp.jpg'
        img.save(image_path)

        result = predict_single_image(image_path)
        highlighted_path, forged_percentage = highlight_forged_area(image_path, image_path)

        output_filename = os.path.basename(highlighted_path)
        output_url = url_for('static', filename=output_filename)

        return jsonify({
            'result': result,
            'forged_area_percent': f"{forged_percentage:.2f}%",
            'highlighted_image_url': request.host_url + output_url
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500


# ✅ Start the Flask server
if __name__ == '__main__':
    print("✅ Flask server is starting...")
    app.run(debug=True)
