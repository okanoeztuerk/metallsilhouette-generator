from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
from math import sqrt
import random
from skimage import morphology
import svgwrite

app = Flask(__name__)

# 1) Einfache Silhouette im "Free Triangle"-Stil
def style_triangle_free(image, step=15, max_side=None, margin=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    h, w = gray.shape
    if max_side is None:
        max_side = step
    canvas = np.zeros((h, w), dtype=np.uint8)
    def tri_height(s): return s * sqrt(3) / 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = gray[y:y+step, x:x+step]
            avg = patch.mean() / 255.0
            side = (1 - avg) * max_side
            if side < 3:
                continue
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
            for (px, py) in pts:
                cv2.circle(canvas, (px, py), 2, 255, -1)
    thick = margin
    cv2.rectangle(canvas,
                  (thick, thick),
                  (w-thick, h-thick),
                  255, thickness=thick)
    return canvas, (h, w)

# Aliase für die ersten sechs Stile nutzen jetzt triangle_free funktion
# 2) Glatte Silhouette als Dreiecke
def glatte_silhouette_tri(image):
    return style_triangle_free(image, step=20, max_side=20, margin=30)
# 3) Einfache Silhouette als Dreiecke
def einfache_silhouette_tri(image):
    return style_triangle_free(image, step=25, max_side=15, margin=30)
# 4) Linienmuster als Dreiecke
def linienmuster_tri(image):
    return style_triangle_free(image, step=20, max_side=25, margin=30)
# 5) Gittermuster als Dreiecke
def gittermuster_tri(image):
    return style_triangle_free(image, step=15, max_side=30, margin=30)
# 6) Organisch als Dreiecke
```python
def organisch_tri(image):
    return style_triangle_free(image, step=18, max_side=22, margin=30)
```# 7) Freie Dreiecke (original)
# (Reuse style_triangle_free)

# Renderer
def render_png(style, data, size, out_png):
    cv2.imwrite(out_png, data)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_png = result_svg = None
    if request.method == 'POST':
        stil = request.form['stil']
        imgfile = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(imgfile, np.uint8), cv2.IMREAD_COLOR)
        os.makedirs('static', exist_ok=True)
        if stil == 'einfach':
            canvas, size = einfache_silhouette_tri(image)
        elif stil == 'glatt':
            canvas, size = glatte_silhouette_tri(image)
        elif stil == 'linien':
            canvas, size = linienmuster_tri(image)
        elif stil == 'gitter':
            canvas, size = gittermuster_tri(image)
        elif stil == 'organisch':
            canvas, size = organisch_tri(image)
        else: # triangle_free
            canvas, size = style_triangle_free(image)
        out_png = 'static/output.png'
        render_png(stil, canvas, size, out_png)
        result_png = out_png
    return render_template('index.html', result_png=result_png)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
