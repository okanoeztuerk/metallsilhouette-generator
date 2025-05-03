from flask import Flask, request, render_template
import cv2
import numpy as np
import os, random
from math import sqrt, cos, sin, pi
import svgwrite

app = Flask(__name__)

# Style‐Parameter für jeden Stil
STYLE_PARAMS = {
    'einfach':       {'step': 8,  'max_side': 8,  'margin': 10},
    'glatt':         {'step': 12, 'max_side': 12, 'margin': 12},
    'linien':        {'step': 20, 'max_side': 1,  'margin': 15},
    'gitter':        {'step': 18, 'max_side': 15, 'margin': 15},
    'organisch':     {'step': 15, 'max_side': 20, 'margin': 20},
    'punktmuster':   {'step': 20, 'max_side': 10, 'margin': 20},
    'triangle_free': {'step': 15, 'max_side': 15, 'margin': 30},
    'vierreck':      {'step': 15, 'max_side': 15, 'margin': 30},  # neue Option
    'vierreck2':      {'step': 15, 'max_side': 15, 'margin': 30},  # neue Option
}

# Farben als BGR-Arrays
COLORS = {
    'schwarz': np.array([  0,   0,   0], dtype=np.uint8),
    'weiss':   np.array([255, 255, 255], dtype=np.uint8),
    'rot':     np.array([  0,   0, 255], dtype=np.uint8),
    'gruen':   np.array([  0, 255,   0], dtype=np.uint8),
    'blau':    np.array([255,   0,   0], dtype=np.uint8),
    'gelb':    np.array([  0, 255, 255], dtype=np.uint8),
    'mint':    np.array([189, 252, 201], dtype=np.uint8),
    'senf':    np.array([  0, 165, 255], dtype=np.uint8),
}

def style_triangle_free(image, step, max_side, color, margin):
    col = (int(color[0]), int(color[1]), int(color[2]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    gray = cv2.bitwise_not(gray)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = gray[y:y+step, x:x+step]
            avg = patch.mean() / 255.0
            side = (1 - avg) * max_side
            if side < 3: continue
            cx = x + step/2 + random.uniform(-step/4, step/4)
            cy = y + step/2 + random.uniform(-step/4, step/4)
            ang = random.uniform(0, 2*pi)
            pts = []
            for i in range(3):
                th = ang + i * 2*pi/3
                px = cx + (side/2) * cos(th)
                py = cy + (side/2) * sin(th)
                pts.append((int(px), int(py)))
            pts_np = np.array(pts, np.int32)
            cv2.fillConvexPoly(mask, pts_np, 255)
            for (px, py) in pts:
                cv2.circle(mask, (px, py), 2, 255, -1)
    cv2.rectangle(mask, (0, 0), (w-1, h-1), 255, thickness=margin*2)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = col
    canvas[mask == 255] = (255,255,255)
    return canvas


def style_density_grid(image, margin, color):
    """
    Teilt das Bild in 2%-Zellen, jede Zelle in 3×3 Sub-Quadrate.
    Helle Zellen → 1 gefülltes Sub-Quadrat, dunkle → bis zu 8.
    margin: Breite des Rahmens (px)
    color: np.array([B, G, R])
    """
    # 0) Farbe in Python-Tuple verwandeln
    col = (int(color[0]), int(color[1]), int(color[2]))

    # 1) Graustufen + Kontrast
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Zellgröße: 2%
    cell_w = max(1, int(w * 0.02))
    cell_h = max(1, int(h * 0.02))
    sub_w = cell_w // 3
    sub_h = cell_h // 3

    # 2) Canvas: farbiger Hintergrund + Rahmen
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:] = col
    # Rahmen in Farbe
    cv2.rectangle(canvas, (0, 0), (w-1, h-1), col, thickness=margin)

    # 3) Sub-Quadrate je nach Helligkeit
    for y0 in range(margin, h-margin, cell_h):
        for x0 in range(margin, w-margin, cell_w):
            patch = gray[y0:y0+cell_h, x0:x0+cell_w]
            if patch.size == 0:
                continue
            avg = patch.mean() / 255.0   # 0..1
            count = int((1 - avg) * 7) + 1   # 1..8

            # alle 9 Sub-Zellen sammeln
            subs = [(x0 + j*sub_w, y0 + i*sub_h) for i in range(3) for j in range(3)]
            fill = random.sample(subs, count)
            for sx, sy in fill:
                cv2.rectangle(
                    canvas,
                    (sx, sy),
                    (sx+sub_w, sy+sub_h),
                    (255,255,255),  # die Quadrate bleiben weiß
                    thickness=-1
                )

    return canvas

def style_rectangle(image, step, max_side, color, margin):
    """
    Erzeugt:
    - farbigen Hintergrund + doppelt dicken Rand,
    - feines Gitter aus weißen Quadraten (Größe ~ avg*max_side),
    - zusätzlich ein Hintergrund-Gitter aus großen weißen Quadraten.
    """
    # 0) Farbe als Tuple
    col = (int(color[0]), int(color[1]), int(color[2]))

    # 1) Graustufen & Kontrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

    h, w = gray.shape

    # 2) Maske für feines Gitter (weiße Kästchen)
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            block = gray[y:y+step, x:x+step]
            avg = block.mean() / 255.0
            side = int(avg * max_side)
            if side < 2:
                continue
            tx = x + (step - side)//2
            ty = y + (step - side)//2
            cv2.rectangle(mask, (tx, ty), (tx+side, ty+side), 255, -1)
    # feines Gitter verbinden
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) Farb-Canvas erstellen
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = col

    # 4) Hintergrund-Gitter aus großen Quadraten
    large_step = step * 5            # z.B. 5-facher Abstand
    large_size = int(max_side * 3)   # z.B. 3-fache Seitenlänge
    for y in range(0, h, large_step):
        for x in range(0, w, large_step):
            # zentriertes großes Quadrat
            gx = x + (large_step - large_size)//2
            gy = y + (large_step - large_size)//2
            if gx < 0 or gy < 0 or gx+large_size > w or gy+large_size > h:
                continue
            cv2.rectangle(canvas,
                          (gx, gy),
                          (gx+large_size, gy+large_size),
                          (255,255,255), -1)

    # 5) feines Gitter (weiße Quadrate) oben drauf
    canvas[mask == 255] = (255,255,255)

    # 6) doppelt dicker, farbiger Rahmen
    cv2.rectangle(canvas, (0,0), (w-1,h-1), col, thickness=margin*2)

    return canvas



def style_triangle_free_svg(image, out_svg_path, step, max_side, margin, color):
    r, g, b = int(color[2]), int(color[1]), int(color[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    gray = cv2.bitwise_not(gray)
    h, w = gray.shape
    dwg = svgwrite.Drawing(out_svg_path, size=(f"{w}px", f"{h}px"))
    dwg.add(dwg.rect(insert=(0,0), size=(w,h),
                     fill=svgwrite.rgb(r,g,b,mode='RGB')))
    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = gray[y:y+step, x:x+step]
            avg = patch.mean()/255.0
            side = (1 - avg)*max_side
            if side < 3: continue
            cx = x + step/2 + random.uniform(-step/4, step/4)
            cy = y + step/2 + random.uniform(-step/4, step/4)
            ang = random.uniform(0, 2*pi)
            pts = []
            for i in range(3):
                th = ang + i * 2*pi/3
                px = cx + (side/2)*cos(th)
                py = cy + (side/2)*sin(th)
                pts.append((px, py))
            dwg.add(dwg.polygon(points=pts, fill='white', stroke='none'))
    dwg.add(dwg.rect(insert=(margin, margin),
                     size=(w-2*margin, h-2*margin),
                     fill='none',
                     stroke=svgwrite.rgb(r,g,b,mode='RGB'),
                     stroke_width=margin*2))
    dwg.save()

@app.route('/', methods=['GET','POST'])
def index():
    upload_url = None
    result_url = None
    result_svg = None
    upload_path = os.path.join('static','upload.png')
    stil  = request.form.get('stil','einfach')
    farbe = request.form.get('farbe','schwarz')

    if request.method == 'POST':
        imgfile = request.files.get('image')
        if imgfile and imgfile.filename:
            buf = imgfile.read()
            image = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            os.makedirs('static', exist_ok=True)
            with open(upload_path,'wb') as f:
                f.write(buf)
        else:
            image = cv2.imread(upload_path)

        upload_url = upload_path
        params = STYLE_PARAMS.get(stil, STYLE_PARAMS['einfach'])
        color  = COLORS.get(farbe, COLORS['schwarz'])

        # Auswahl der richtigen Render-Funktion
        if stil == 'vierreck':
            canvas = style_rectangle(
                image,
                step=params['step'],
                max_side=params['max_side'],
                color=color,
                margin=params['margin']
            )
        elif stil == 'vierreck2':
            canvas = style_density_grid(
                image,
                margin=params['margin'],
                color=color
            )
        else:
            canvas = style_triangle_free(
                image,
                step=params['step'],
                max_side=params['max_side'],
                color=color,
                margin=params['margin']
            )

        # PNG speichern
        out_png = os.path.join('static','output.png')
        cv2.imwrite(out_png, canvas)
        result_url = out_png

        # SVG nur für triangle_free
        if stil == 'triangle_free':
            svg_p = os.path.join('static','output.svg')
            style_triangle_free_svg(
                image,
                svg_p,
                step=params['step'],
                max_side=params['max_side'],
                margin=params['margin'],
                color=color
            )
            result_svg = svg_p

    return render_template('index.html',
                           upload_url=upload_url,
                           result_url=result_url,
                           result_svg=result_svg,
                           stil=stil,
                           farbe=farbe)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))
