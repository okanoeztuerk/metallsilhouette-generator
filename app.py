from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from skimage import measure, morphology
import svgwrite

app = Flask(__name__)

def einfache_silhouette(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image.shape[:2]

def glatte_silhouette(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 10)
    cleaned = morphology.remove_small_objects(adaptive.astype(bool), min_size=500)
    cleaned = (cleaned * 255).astype(np.uint8)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, image.shape[:2]

def punktmuster(image):
    # Halbton-Effekt mit Floyd–Steinberg Dithering
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Otsu-Schwellenwert
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dithering
    for y in range(bw.shape[0]):
        for x in range(bw.shape[1]):
            old = bw[y, x]
            new = 255 if old > 128 else 0
            bw[y, x] = new
            err = old - new
            if x+1 < bw.shape[1]: bw[y, x+1] = np.clip(bw[y, x+1] + err*7/16, 0, 255)
            if y+1 < bw.shape[0]:
                if x>0: bw[y+1, x-1] = np.clip(bw[y+1, x-1] + err*3/16, 0,255)
                bw[y+1, x]   = np.clip(bw[y+1, x]   + err*5/16, 0,255)
                if x+1< bw.shape[1]: bw[y+1, x+1] = np.clip(bw[y+1, x+1]+err*1/16,0,255)
    # Erstelle Punktmuster-Kreise
    contours = []
    h, w = bw.shape
    step = 10  # Rasterabstand
    for y in range(0, h, step):
        for x in range(0, w, step):
            val = bw[y, x]
            if val==0:
                r = int(step/2)
            else:
                r = int((1 - val/255) * (step/2))
            contours.append([(x, y, r)])
    return contours, (h, w)

def linienmuster(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50,150)
    # Distance transform
    dist = cv2.distanceTransform(255-edges, cv2.DIST_L2, 5)
    # Linien im Abstand
    contours = []
    h, w = gray.shape
    spacing = 15
    for i in range(0, h, spacing):
        points = [(0,i), (w,i)]
        contours.append(points)
    return contours, (h, w)

def render_png(contours, size, out_png, style):
    h, w = size
    canvas = np.ones((h, w), np.uint8) * 255
    if style in ("einfach","glatt"):
        cnts = np.array(contours, dtype=object)
        cv2.drawContours(canvas, cnts, -1, (0), thickness=cv2.FILLED)
    elif style=="punktmuster":
        for x,y,r in contours:
            cv2.circle(canvas,(x,y),r,(0),-1)
    elif style=="linien":
        for p in contours:
            cv2.line(canvas,p[0],p[1],0,1)
    # Rahmen
    cv2.rectangle(canvas,(10,10),(w-10,h-10),0,2)
    cv2.imwrite(out_png,canvas)

def render_svg(contours, size, out_svg, style):
    h, w = size
    dwg = svgwrite.Drawing(out_svg, size=(f"{w}px",f"{h}px"))
    if style in ("einfach","glatt"):
        for cnt in contours:
            pts=[(int(p[0][0]),int(p[0][1])) for p in cnt]
            dwg.add(dwg.polygon(pts, fill='black'))
    elif style=="punktmuster":
        for x,y,r in contours:
            dwg.add(dwg.circle(center=(x,y), r=r, fill='black'))
    elif style=="linien":
        for p in contours:
            dwg.add(dwg.line(start=p[0], end=p[1], stroke='black', stroke_width=1))
    dwg.add(dwg.rect(insert=(10,10), size=(w-20,h-20), fill='none',
                     stroke='black', stroke_width=2))
    dwg.save()

@app.route('/', methods=['GET','POST'])
def index():
    result_url=svg_url=stil=None
    if request.method=='POST':
        stil = request.form['stil']
        imgfile = request.files['image']
        image = cv2.imdecode(np.frombuffer(imgfile.read(), np.uint8),cv2.IMREAD_COLOR)
        if stil in ("einfach","glatt"):
            contours, size = (glatte_silhouette(image) if stil=="glatt"
                              else einfache_silhouette(image))
        elif stil=="punktmuster":
            contours, size = punktmuster(image)
        elif stil=="linien":
            contours, size = linienmuster(image)
        # PNG
        result_url = os.path.join('static','output.png')
        render_png(contours, size, result_url, stil)
        # SVG
        svg_url = os.path.join('static','output.svg')
        render_svg(contours, size, svg_url, stil)
    return render_template('index.html',
                           result_url=result_url,
                           svg_url=svg_url,
                           stil=stil)
