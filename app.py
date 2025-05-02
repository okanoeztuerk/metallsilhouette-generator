from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
from math import sqrt
import random
from skimage import morphology
import svgwrite

app = Flask(__name__)

# 1) Einfache Silhouette
# ... ältere Funktionen bleiben unverändert

def einfache_silhouette(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image.shape[:2]

# 2) Glatte Silhouette
# ...

def glatte_silhouette(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 10)
    cleaned = morphology.remove_small_objects(adaptive.astype(bool), min_size=500)
    cleaned = (cleaned * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image.shape[:2]

# ... weitere Stile: linienmuster, gittermuster, organisch ...

# NEUE FUKTION: Freie Dreiecke mit Knotenpunkten (invertiert)
def triangle_free_nodes(image, step=15, max_side=None, margin=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # optional Kontrast
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    h, w = gray.shape
    if max_side is None:
        max_side = step
    # Hilfsfunktion Höhe
    def tri_height(s): return s * sqrt(3) / 2

    # schwarzer Canvas
    canvas = np.zeros((h, w), dtype=np.uint8)
    # Raster
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = gray[y:y+step, x:x+step]
            avg = patch.mean() / 255.0
            side = (1 - avg) * max_side
            if side < 3:
                continue
            # Jitter und freier Winkel
            cx = x + step/2 + random.uniform(-step/4, step/4)
            cy = y + step/2 + random.uniform(-step/4, step/4)
            angle = random.uniform(0, 2*np.pi)
            pts = []
            for i in range(3):
                theta = angle + i * 2 * np.pi/3
                px = cx + (side/2) * np.cos(theta)
                py = cy + (side/2) * np.sin(theta)
                pts.append((int(px), int(py)))
            pts_np = np.array(pts, np.int32)
            cv2.fillConvexPoly(canvas, pts_np, 255)
            # Knoten
            for (px, py) in pts:
                cv2.circle(canvas, (px, py), 2, 255, -1)

    # Rahmen
    thick = margin
    cv2.rectangle(canvas,
                  (thick, thick),
                  (w-thick, h-thick),
                  255, thickness=thick)
    return canvas, (h, w)

# Renderer

def render_png(style, data, size, out_png):
    h, w = size
    if style == 'triangle_free':
        # data ist Canvas
        cv2.imwrite(out_png, data)
        return
    # andere Stile unverändert ...

@app.route('/', methods=['GET', 'POST'])
def index():
    result_png = result_svg = None
    if request.method == 'POST':
        stil = request.form['stil']
        imgfile = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(imgfile, np.uint8), cv2.IMREAD_COLOR)
        if stil == 'triangle_free':
            canvas, size = triangle_free_nodes(image)
            os.makedirs('static', exist_ok=True)
            out_png = 'static/output.png'
            render_png(stil, canvas, size, out_png)
            result_png = out_png
            result_svg = None
        else:
            # bestehende Logik für andere Stile
            pass
    return render_template('index.html', result_png=result_png, result_svg=result_svg)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
