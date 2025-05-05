import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageColor, ImageFilter
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
    _, labels, centers = cv2.kmeans(pixels, 8, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_img = labels.flatten().reshape((image_resized.shape[:2]))

    mask = np.zeros_like(segmented_img, dtype=np.uint8)
    for i, c in enumerate(centers):
        l, a, b = c
        if 100 < l < 200 and 115 < a < 145 and 115 < b < 145:
            mask[segmented_img == i] = 255

    h, w = mask.shape
    coords = np.argwhere(mask == 255)
    np.random.shuffle(coords)
    occupied = np.zeros((h, w), dtype=bool)

    bg_color = request.form.get("color", "#98ffcc")
    width_cm = float(request.form.get("width_cm", "30"))
    aspect_ratio = w / h
    height_cm = round(width_cm / aspect_ratio, 1)
    price = round(width_cm * height_cm * 0.15, 2)

    img = Image.new("RGBA", (w, h), (*ImageColor.getrgb(bg_color), 255))
    draw = ImageDraw.Draw(img)
    count = 0
    for y, x in coords:
        if count > 1500:
            break
        radius = np.random.randint(3, 7)
        buffer = radius + 2
        x1, y1, x2, y2 = x - buffer, y - buffer, x + buffer, y + buffer
        if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
            continue
        if not occupied[y1:y2, x1:x2].any():
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, 0))
            occupied[y1:y2, x1:x2] = True
            count += 1

    border_thickness = 15
    draw.rectangle([0, 0, w - 1, h - 1], outline=ImageColor.getrgb(bg_color), width=border_thickness)
    img.save("static/output.png")

    dwg = svgwrite.Drawing("static/output.svg", size=(w, h))
    count = 0
    for y, x in coords:
        if count > 1500:
            break
        radius = np.random.randint(3, 7)
        buffer = radius + 2
        x1, y1, x2, y2 = x - buffer, y - buffer, x + buffer, y + buffer
        if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
            continue
        if not occupied[y1:y2, x1:x2].any():
            dwg.add(dwg.circle(center=(float(x), float(y)), r=radius, fill='none', stroke='black', stroke_width=0.1))
            occupied[y1:y2, x1:x2] = True
            count += 1
    dwg.save()

    def create_preview(generated_path, background_path, width_cm_real, preview_path):
        background = Image.open(background_path).convert("RGBA")
        foreground = Image.open(generated_path).convert("RGBA")
        bg_w, bg_h = background.size
        wand_pixel_breite = int(bg_w * 0.75)
        pixel_per_cm = wand_pixel_breite / 200.0
        new_width_px = int(width_cm_real * pixel_per_cm)
        ratio = foreground.width / foreground.height
        new_height_px = int(new_width_px / ratio)
        foreground_resized = foreground.resize((new_width_px, new_height_px), Image.LANCZOS)

        pos_x = (bg_w - new_width_px) // 2
        sofa_unterkante_y = int(bg_h * 0.5)  # früher war 0.7
        pos_y = sofa_unterkante_y - new_height_px - 20

        
        thickness = 6
        
        # Farbe des sichtbaren Rahmens (Mintfarbe)
        draw_visible = ImageDraw.Draw(foreground_resized)
        draw_visible.rectangle([0, 0, foreground_resized.width - 1, foreground_resized.height - 1], outline=ImageColor.getrgb(bg_color), width=thickness)


        background.paste(foreground_resized, (pos_x, pos_y), foreground_resized)
        background.convert("RGB").save(preview_path)

    create_preview("static/output.png", "static/background.jpg", width_cm, "static/preview.png")

    return render_template("index.html", result=True,
                           width_cm=width_cm,
                           height_cm=height_cm,
                           price=price)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)







