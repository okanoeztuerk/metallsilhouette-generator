from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from math import sqrt, cos, sin, pi
from skimage import morphology
import svgwrite

app = Flask(__name__)

# 1) Einfache Silhouette
def einfache_silhouette(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image.shape[:2]

# 2) Glatte Silhouette
def glatte_silhouette(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 10)
    cleaned = morphology.remove_small_objects(adaptive.astype(bool), min_size=500)
    cleaned = (cleaned * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image.shape[:2]

# 3) Punktmuster → jetzt als invertiertes, feines Waben-Lochmuster
def punktmuster(image, step=20, max_side=10, margin=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    def hexagon(center, side):
        x0, y0 = center
        pts = []
        for i in range(6):
            angle = pi/6 + i * pi/3
            x = x0 + side * cos(angle)
            y = y0 + side * sin(angle)
            pts.append([int(x), int(y)])
        return np.array(pts, np.int32)

    # schwarzer Hintergrund
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    row_h = sqrt(3) * step / 2
    for row in range(int((h - 2*margin) // row_h) + 2):
        y = int(margin + row * row_h)
        offset = 0 if row % 2 == 0 else step/2
        for col in range(int((w - 2*margin) // step) + 2):
            x = int(margin + col * step + offset)
            patch = gray[
                max(0, y-step//2):min(h, y+step//2),
                max(0, x-step//2):min(w, x+step//2)
            ]
            avg = patch.mean() / 255.0
            side = (1 - avg) * max_side
            if side < 1:
                continue
            pts = hexagon((x, y), side)
            # weiße Wabe (Loch)
            cv2.fillPoly(canvas, [pts], (255, 255, 255))

    # weißer Rahmen
    cv2.rectangle(canvas,
                  (margin, margin),
                  (w-margin, h-margin),
                  (255, 255, 255),
                  thickness=2)
    return canvas, (h, w)

# 4) Linienmuster
def linienmuster(image, spacing=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    contours = []
    for y in range(0, h, spacing):
        contours.append(((0, y), (w, y)))
    return contours, (h, w)

# 5) Geometrisches Gitter (Quadrate)
def gittermuster(image, size=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    shapes = []
    for y in range(0, h, size):
        for x in range(0, w, size):
            block = gray[y:y+size, x:x+size]
            mean = block.mean() / 255.0
            if mean < 0.5:
                shapes.append([(x,y), (x+size,y), (x+size,y+size), (x,y+size)])
    return shapes, (h, w)

# 6) Organische Ausschnitte
def organisch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth = [cv2.approxPolyDP(c, epsilon=5, closed=True) for c in cnts]
    return smooth, image.shape[:2]

# Renderer für PNG
def render_png_from_canvas(canvas, out_png):
    cv2.imwrite(out_png, canvas)

# Renderer für Konturen/Shapes
def render_contours_png(contours, size, out_png):
    h, w = size
    img = np.ones((h, w, 3), np.uint8) * 255
    cv2.drawContours(img, np.array(contours, dtype=object), -1, (0,0,0), thickness=cv2.FILLED)
    cv2.rectangle(img,(10,10),(w-10,h-10),(0,0,0),2)
    cv2.imwrite(out_png, img)

# Renderer für SVG
def render_svg(contours_or_canvas, size, out_svg, style):
    w, h = size[1], size[0]
    dwg = svgwrite.Drawing(out_svg, size=(f"{w}px", f"{h}px"))
    if style == "punktmuster":
        # contours_or_canvas is actually full canvas mask
        # we rasterize canvas to svg: invert pixels to paths is complex; skip SVG for this style
        pass
    else:
        for shape in contours_or_canvas:
            if isinstance(shape[0], tuple):
                # lines
                dwg.add(dwg.line(start=shape[0], end=shape[1], stroke='black', stroke_width=1))
            else:
                pts = [(int(p[0][0]), int(p[0][1])) for p in shape]
                dwg.add(dwg.polygon(pts, fill='black'))
        dwg.add(dwg.rect(insert=(10,10),
                         size=(w-20,h-20),
                         fill='none', stroke='black', stroke_width=2))
        dwg.save()

@app.route('/', methods=['GET', 'POST'])
def index():
    result_png = result_svg = None
    if request.method == 'POST':
        stil = request.form['stil']
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Stil-Auswahl
        if stil == 'einfach':
            contours, size = einfache_silhouette(image)
            render_contours_png(contours, size, 'static/output.png')
            render_svg(contours, size, 'static/output.svg', stil)

        elif stil == 'glatt':
            contours, size = glatte_silhouette(image)
            render_contours_png(contours, size, 'static/output.png')
            render_svg(contours, size, 'static/output.svg', stil)

        elif stil == 'punktmuster':
            canvas, size = punktmuster(image, step=20, max_side=10, margin=50)
            render_png_from_canvas(canvas, 'static/output.png')
            # SVG für Lochmuster üblicherweise aus Metallschnitt-Software exportieren
            # wir liefern nur PNG-Vorschau hier
            result_svg = None

        elif stil == 'linien':
            contours, size = linienmuster(image)
            render_contours_png(contours, size, 'static/output.png')
            render_svg(contours, size, 'static/output.svg', stil)

        elif stil == 'gitter':
            shapes, size = gittermuster(image)
            render_contours_png(shapes, size, 'static/output.png')
            render_svg(shapes, size, 'static/output.svg', stil)

        elif stil == 'organisch':
            contours, size = organisch(image)
            render_contours_png(contours, size, 'static/output.png')
            render_svg(contours, size, 'static/output.svg', stil)

        result_png = 'static/output.png'
        if result_svg is None:
            result_svg = '— SVG nur in PNG-korrigier-Software —'
        else:
            result_svg = 'static/output.svg'

    return render_template('index.html',
                           result_png=result_png,
                           result_svg=result_svg)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
