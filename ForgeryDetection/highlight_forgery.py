import cv2
import numpy as np
import os

# Create output directory
os.makedirs("static", exist_ok=True)

def highlight_forged_area(real_path, forged_path):
    print("🔍 Loading images...")
    real = cv2.imread(real_path)
    forged = cv2.imread(forged_path)

    if real is None or forged is None:
        print("❌ Could not load one or both images.")
        return None, 0

    # Resize both to same size
    real = cv2.resize(real, (256, 256))
    forged = cv2.resize(forged, (256, 256))

    # Convert to grayscale
    gray_real = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    gray_forged = cv2.cvtColor(forged, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    gray_real = cv2.GaussianBlur(gray_real, (5, 5), 0)
    gray_forged = cv2.GaussianBlur(gray_forged, (5, 5), 0)

    print("✅ Calculating absolute difference...")
    diff = cv2.absdiff(gray_real, gray_forged)

    # Threshold to extract regions of change
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Optional: morphological operations
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = forged.copy()
    forged_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Skip noise
            continue
        x, y, w, h = cv2.boundingRect(contour)
        forged_area += area
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    percent_forged = (forged_area / (256 * 256)) * 100

    # Save the highlighted output
    output_path = "static/highlighted_forged.jpg"
    cv2.imwrite(output_path, result)

    # Also save a heatmap visualization
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(forged, 0.6, heatmap, 0.4, 0)
    cv2.imwrite("static/diff_heatmap.jpg", blended)

    print("💾 Saved: highlighted_forged.jpg and diff_heatmap.jpg")
    print(f"📊 Forged Area: {percent_forged:.2f}%")

    return output_path, round(percent_forged, 2)
