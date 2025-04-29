from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from skimage import measure
import svgwrite

app = Flask(__name__)

def contours_to_svg(edge_img, svg_path, scale=1.0):
    contours = measure.find_contours(edge_img, 0.8)
    all_points = np.vstack(contours)
    min_y, min_x = np.min(all_points, axis=0)
    max_y, max_x = np.max(all_points, axis=0)
    width_px = max_x - min_x
    height_px = max_y - min_y

    svg_width_cm = 60.0  # Standardbreite in cm
    px_to_cm = svg_width_cm / width_px
    svg_height_cm = height_px * px_to_cm

    dwg = svgwrite.Drawing(svg_path, size=(f"{svg_width_cm}cm", f"{svg_height_cm}cm"))
    for contour in contours:
        points = [(p[1] * px_to_cm, p[0] * px_to_cm) for p in contour]
        dwg.add(dwg.polyline(points=points, fill='black', stroke='none'))
    return svg_width_cm, svg_height_cm

@app.route('/', methods=['GET', 'POST'])
def index():
    preis = None
    breite_cm = None
    höhe_cm = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(img, 50, 150)

            output_path = os.path.join('static', 'silhouette.png')
            cv2.imwrite(output_path, edges)

            svg_path = os.path.join('static', 'silhouette.svg')
            breite_cm, höhe_cm = contours_to_svg(edges, svg_path)

            fläche = breite_cm * höhe_cm
            preis = round(20.0 + (fläche * 0.08), 2)

            return render_template('index.html',
                                   result_url=output_path,
                                   svg_url=svg_path,
                                   preis=preis,
                                   breite_cm=round(breite_cm, 2),
                                   höhe_cm=round(höhe_cm, 2))
    return render_template('index.html')
