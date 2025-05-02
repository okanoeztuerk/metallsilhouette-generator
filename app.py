from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from math import sqrt, cos, sin, pi
import random
from skimage import morphology

app = Flask(__name__)

# Helper: equilateral triangle height
def tri_height(side):
    return side * sqrt(3) / 2

def style_triangle_free(image, step, max_side, color, margin):
    # color: np.array([B,G,R])
    col = (int(color[0]), int(color[1]), int(color[2]))

    # 1) Graustufe + Kontrast + Inversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    gray = cv2.bitwise_not(gray)

    h, w = gray.shape

    # 2) Maske: Dreiecke + Rahmen als WEIß (255) auf SCHWARZ (0)
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = gray[y:y+step, x:x+step]
            avg = patch.mean() / 255.0
            side = (1 - avg) * max_side
            if side < 3:
                continue

            # Zufalls-Jitter & Eckpunkte
            cx = x + step/2 + random.uniform(-step/4, step/4)
            cy = y + step/2 + random.uniform(-step/4, step/4)
            angle0 = random.uniform(0, 2*pi)
            pts = []
            for i in range(3):
                theta = angle0 + i * 2*pi/3
                px = cx + (side/2) * cos(theta)
                py = cy + (side/2) * sin(theta)
                pts.append((int(px), int(py)))

            pts_np = np.array(pts, np.int32)
            cv2.fillConvexPoly(mask, pts_np, 255)
            for (px, py) in pts:
                cv2.circle(mask, (px, py), 2, 255, -1)

    # Rahmen in der Maske
    cv2.rectangle(mask, (0, 0), (w-1, h-1), 255, thickness=margin)

    # 3) Farb-Canvas: komplett mit der Wunschfarbe füllen
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = col  # Hintergrund & Rahmenfarbe

    # 4) Dreiecke in Weiß setzen
    canvas[mask == 255] = (255, 255, 255)

    return canvas




# Style parameters for each option
STYLE_PARAMS = {
    'einfach':      {'step':8,  'max_side':8,  'margin':10},
    'glatt':        {'step':12, 'max_side':12, 'margin':12},
    'linien':       {'step':20, 'max_side':1,  'margin':15},
    'gitter':       {'step':18, 'max_side':15, 'margin':15},
    'organisch':    {'step':15, 'max_side':20, 'margin':20},
    'punktmuster':  {'step':20, 'max_side':10, 'margin':20},
    'triangle_free':{'step':15, 'max_side':15, 'margin':30},
}

# 6 selectable colors (BGR for OpenCV)
COLORS = {
    'schwarz': np.array([  0,   0,   0], dtype=np.uint8),
    'weiss':   np.array([255, 255, 255], dtype=np.uint8),
    'rot':     np.array([  0,   0, 255], dtype=np.uint8),
    'gruen':   np.array([  0, 255,   0], dtype=np.uint8),
    'blau':    np.array([255,   0,   0], dtype=np.uint8),
    'gelb':    np.array([  0, 255, 255], dtype=np.uint8),
    'mint':   np.array([189, 252, 201], dtype=np.uint8),  # helle Mint-Farbe
    'senf':   np.array([  0, 165, 255], dtype=np.uint8),  # Senfgelb
}

@app.route('/', methods=['GET','POST'])
def index():
    result_url = None
    upload_url = None

    if request.method == 'POST':
        imgfile = request.files['image']
        buf = imgfile.read()
        image = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

        # save upload for preview
        os.makedirs('static', exist_ok=True)
        upload_path = os.path.join('static','upload.png')
        with open(upload_path, 'wb') as f:
            f.write(buf)
        upload_url = upload_path

        stil  = request.form['stil']
        farbe = request.form['farbe']
        params = STYLE_PARAMS.get(stil, STYLE_PARAMS['einfach'])
        color  = COLORS.get(farbe, COLORS['schwarz'])

        canvas = style_triangle_free(
            image,
            step=params['step'],
            max_side=params['max_side'],
            color=color,
            margin=params['margin']
        )

        out_path = os.path.join('static','output.png')
        cv2.imwrite(out_path, canvas)
        result_url = out_path

    return render_template(
        'index.html',
        upload_url=upload_url,
        result_url=result_url
    )

if __name__=='__main__':
    port = int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0', port=port)
