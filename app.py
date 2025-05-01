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

# 3) Punktmuster → invertiertes, feines Waben-Lochmuster
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
            if side < 1: continue
            pts = hexagon((x, y), side)
            cv2.fillPoly(canvas, [pts], (255,255,255))

    # weißer Rahmen
    cv2.rectangle(canvas, (margin, margin), (w-margin, h-margin), (255,255,255), 2)
    return canvas, (h, w)

# 4) Linienmuster
def linienmuster(image, spacing=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lines = [((0, y), (w, y)) for y in range(0, h, spacing)]
    return lines, (h, w)

# 5) Gittermuster (Quadrate)
def gittermuster(image, size=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    quads = []
    for y in range(0, h, size):
        for x in range(0, w, size):
            block = gray[y:y+size, x:x+size]
            if block.mean()/255 < 0.5:
                quads.append(np.array([[x,y],[x+size,y],[x+size,y+size],[x,y+size]], np.int32))
    return quads, (h, w)

# 6) Organische Ausschnitte
def organisch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth = [cv2.approxPolyDP(c, epsilon=5, closed=True) for c in cnts]
    return smooth, image.shape[:2]


# Renderer für alle Stile
def render_png(style, data, size, out_png):
    h, w = size
    if style == 'punktmuster':
        # data ist ein Canvas
        cv2.imwrite(out_png, data)
        return

    # sonst: Start mit weißem Hintergrund
    canvas = np.ones((h, w, 3), np.uint8) * 255

    if style in ('einfach','glatt','organisch'):
        # data = contours
        cv2.drawContours(canvas, data, -1, (0,0,0), thickness=cv2.FILLED)
    elif style == 'linien':
        # data = list of lines
        for (p1, p2) in data:
            cv2.line(canvas, p1, p2, (0,0,0), 1)
    elif style == 'gitter':
        # data = list of quads (np arrays)
        for quad in data:
            cv2.fillPoly(canvas, [quad], (0,0,0))

    # Rahmen um Bild
    cv2.rectangle(canvas, (10,10), (w-10,h-10), (0,0,0), 2)
    cv2.imwrite(out_png, canvas)


@app.route('/', methods=['GET','POST'])
def index():
    result_png = result_svg = None

    if request.method == 'POST':
        stil = request.form['stil']
        imgfile = request.files['image']
        image = cv2.imdecode(np.frombuffer(imgfile.read(), np.uint8), cv2.IMREAD_COLOR)

        if stil in ('einfach','glatt','linien','gitter','organisch'):
            fn = {
                'einfach': einfache_silhouette,
                'glatt': glatte_silhouette,
                'linien': linienmuster,
                'gitter': gittermuster,
                'organisch': organisch
            }[stil]
            data, size = fn(image)
        else:
            # punktmuster
            data, size = punktmuster(image, step=20, max_side=10, margin=50)

        # PNG rendern
        os.makedirs('static', exist_ok=True)
        out_png = 'static/output.png'
        render_png(stil, data, size, out_png)
        result_png = out_png

        # SVG: nur für Kontur-Stile, nicht für punktmuster
        if stil != 'punktmuster':
            # Einfach Konturen in SVG (für gitter, organisch, einfach, glatt, linien)
            out_svg = 'static/output.svg'
            dwg = svgwrite.Drawing(out_svg, size=(f"{size[1]}px", f"{size[0]}px"))
            if stil in ('einfach','glatt','organisch'):
                for cnt in data:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    dwg.add(dwg.polygon(pts, fill='black'))
            elif stil == 'linien':
                for p1,p2 in data:
                    dwg.add(dwg.line(start=p1, end=p2, stroke='black', stroke_width=1))
            elif stil == 'gitter':
                for quad in data:
                    pts = [(int(x),int(y)) for [x,y] in quad]
                    dwg.add(dwg.polygon(pts, fill='black'))
            dwg.add(dwg.rect(insert=(10,10), size=(size[1]-20, size[0]-20),
                             fill='none', stroke='black', stroke_width=2))
            dwg.save()
            result_svg = out_svg

    return render_template('index.html',
                           result_png=result_png,
                           result_svg=result_svg,
                           stil=stil)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0', port=port)
