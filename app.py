
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageDraw
import svgwrite

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=False)

@app.route("/generate", methods=["POST"])
def generate():
    file = request.files["image"]
    path = "static/upload.jpg"
    file.save(path)

    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, int(image_rgb.shape[0] * 512 / image_rgb.shape[1])))
    lab = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_img = labels.flatten().reshape((image_resized.shape[:2]))

    mask = np.zeros_like(segmented_img, dtype=np.uint8)
    for i, c in enumerate(centers):
        l, a, b = c
        if 100 < l < 200 and 115 < a < 145 and 115 < b < 145:
            mask[segmented_img == i] = 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)
    occupied = np.zeros((h, w), dtype=bool)

    for cnt in contours:
        for i, pt in enumerate(cnt[::2]):
            x, y = pt[0]
            dx, dy = x - w // 2, y - h // 2
            dist = np.sqrt(dx**2 + dy**2)
            norm = dist / np.sqrt((w // 2)**2 + (h // 2)**2)
            r = int(1 + (1 - norm) * 3)
            s = r + 1
            if 0 <= x-s < w and 0 <= y-s < h and x+s < w and y+s < h:
                if not occupied[y-s:y+s, x-s:x+s].any():
                    draw.ellipse((x - r, y - r, x + r, y + r), fill="white")
                    occupied[y-s:y+s, x-s:x+s] = True

    coords = np.argwhere(mask == 255)
    np.random.shuffle(coords)
    count = 0
    for y, x in coords:
        if count > 4000: break
        r = np.random.randint(1, 4)
        s = r + 1
        if 0 <= x-s < w and 0 <= y-s < h and x+s < w and y+s < h:
            if not occupied[y-s:y+s, x-s:x+s].any():
                draw.ellipse((x - r, y - r, x + r, y + r), fill="white")
                occupied[y-s:y+s, x-s:x+s] = True
                count += 1

    img.save("static/output.png")

    # SVG schreiben
    svg_path = "static/output.svg"
    dwg = svgwrite.Drawing(svg_path, size=(w, h), profile='tiny')
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill='black'))
    px = np.array(img)
    ys, xs = np.where(np.all(px == [255, 255, 255], axis=-1))
    for x, y in zip(xs, ys):
        dwg.add(dwg.circle(center=(float(x), float(y)), r=1.2, fill='white'))
    dwg.save()

    return render_template("index.html", result=True)

if __name__ == "__main__":
    app.run(debug=True)
