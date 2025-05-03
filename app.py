from flask import Flask, request, render_template
import cv2, numpy as np, os, random, svgwrite
from math import sqrt, cos, sin, pi
from skimage import morphology

app = Flask(__name__)

# Helper für Dreieckshöhe
def tri_height(side):
    return side * sqrt(3) / 2

def style_triangle_free(image, step, max_side, color, margin):
    # Farbe als Tuple
    col = (int(color[0]), int(color[1]), int(color[2]))

    # Graustufen + Kontrast + Inversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    gray = cv2.bitwise_not(gray)

    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)

    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = gray[y:y+step, x:x+step]
            avg = patch.mean()/255.0
            side = (1-avg)*max_side
            if side < 3: continue

            cx = x+step/2 + random.uniform(-step/4, step/4)
            cy = y+step/2 + random.uniform(-step/4, step/4)
            ang = random.uniform(0, 2*pi)
            pts = []
            for i in range(3):
                th = ang + i*2*pi/3
                px = cx + (side/2)*cos(th)
                py = cy + (side/2)*sin(th)
                pts.append((int(px), int(py)))
            pts_np = np.array(pts, np.int32)
            cv2.fillConvexPoly(mask, pts_np, 255)
            for (px, py) in pts:
                cv2.circle(mask, (px, py), 2, 255, -1)

    # Rahmen als weiße Maske
    cv2.rectangle(mask, (0,0), (w-1,h-1), 255, thickness=margin*2)

    # Farb-Canvas
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:] = col
    # Dreiecke weiß
    canvas[mask==255] = (255,255,255)
    return canvas

# Stile & Farben unverändert ...

@app.route('/', methods=['GET','POST'])
def index():
    upload_url = None
    result_url = None
    result_svg = None
    stil = request.form.get('stil','einfach')
    farbe = request.form.get('farbe','schwarz')

    # Pfad für persistentes Upload
    upload_path = os.path.join('static','upload.png')

    if request.method=='POST':
        imgfile = request.files.get('image')
        # wenn neue Datei hochgeladen wurde
        if imgfile and imgfile.filename:
            buf = imgfile.read()
            image = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            os.makedirs('static', exist_ok=True)
            with open(upload_path,'wb') as f:
                f.write(buf)
        else:
            # kein neuer Upload → nutze bestehende
            image = cv2.imread(upload_path)

        upload_url = upload_path
        params = STYLE_PARAMS.get(stil, STYLE_PARAMS['einfach'])
        color  = COLORS.get(farbe, COLORS['schwarz'])

        # PNG
        canvas = style_triangle_free(
            image,
            step=params['step'],
            max_side=params['max_side'],
            color=color,
            margin=params['margin']
        )
        out_png = os.path.join('static','output.png')
        cv2.imwrite(out_png, canvas)
        result_url = out_png

        # SVG
        svg_p = os.path.join('static','output.svg')
        style_triangle_free_svg(
            image, svg_p,
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

if __name__=='__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)))
