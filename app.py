import os
import cv2
import numpy as np
from flask import Flask, render_template, request, session, jsonify
from PIL import Image, ImageDraw, ImageColor, ImageFilter
import svgwrite

import uuid
app = Flask(__name__)
app.secret_key = "supersecretkey"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=False, image_uploaded=os.path.exists("static/upload.jpg"))


@app.route("/widget")
def widget():
    return render_template("widget.html")


@app.route("/api/generate-shopify", methods=["POST"])
def generate_shopify():
    generate()
    base_url = request.url_root.rstrip("/")
    return jsonify({
        "type": "wandbild_ready",
        "output_preview_url": f"{base_url}/static/preview.png",
        "output_png_url": f"{base_url}/static/output.png",
        "output_svg_url": f"{base_url}/static/output.svg"
    })


@app.route("/generate", methods=["POST"])
def generate():
    if "image" in request.files and request.files["image"].filename:
        file = request.files["image"]
        path = "static/upload.jpg"
        file.save(path)
        session['image_uploaded'] = True
    else:
        if not os.path.exists("static/upload.jpg"):
            return render_template("index.html", result=False, error="Kein Bild vorhanden.")
        path = "static/upload.jpg"

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
    bg_color = request.form.get("color", "#98ffcc")
    width_cm = float(request.form.get("width_cm", "100"))  # default now 100 cm
    aspect_ratio = w / h
    height_cm = round(width_cm / aspect_ratio, 1)
    price = round(width_cm * height_cm * 0.15 * 0.5, 2)

    shape_type = request.form.get("shape", "circle")

    coords = np.argwhere(mask == 255)
    np.random.shuffle(coords)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = Image.new("RGBA", (w, h), (*ImageColor.getrgb(bg_color), 255))
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
                    if shape_type == "circle":
                        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 0))
                    elif shape_type == "square":
                        draw.rectangle((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 0))
                    elif shape_type == "triangle":
                        draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=(255, 255, 255, 0))
                    elif shape_type == "sand":
                        heart = Image.new("L", (2*r+2, 2*r+2), 0)
                        d = ImageDraw.Draw(heart)
                        d.polygon([(r, 0), (0, r), (2*r, r), (r, 2*r)], fill=255)
                        img.paste(Image.new("RGBA", heart.size, (255, 255, 255, 0)), (x - r, y - r), heart)
                    elif shape_type == "realHeart":
                        heart = Image.new("L", (2*r+4, 2*r+4), 0)
                        hd = ImageDraw.Draw(heart)
                        hd.polygon([
                            (r+2, r//2), (r//2, 0), (0, r//2), (r, 2*r),
                            (2*r, r//2), (3*r//2, 0), (r+2, r//2)
                        ], fill=255)
                        img.paste(Image.new("RGBA", heart.size, (255, 255, 255, 0)), (x - r, y - r), heart)
                    elif shape_type == "S":
                        s_path = Image.new("L", (2*r+4, 3*r+4), 0)
                        d = ImageDraw.Draw(s_path)
                        d.arc([0, 0, 2*r, 2*r], start=0, end=180, fill=255)
                        d.arc([0, r, 2*r, 3*r], start=180, end=360, fill=255)
                        img.paste(Image.new("RGBA", s_path.size, (255, 255, 255, 0)), (x - r, y - r), s_path)
                    elif shape_type == "I":
                        draw.rectangle((x - r//3, y - r, x + r//3, y + r), fill=(255, 255, 255, 0))
                    occupied[y-s:y+s, x-s:x+s] = True

    count = 0
    for y, x in coords:
        if count > 2000:
            break
        r = np.random.randint(1, 4)
        s = r + 1
        if 0 <= x-s < w and 0 <= y-s < h and x+s < w and y+s < h:
            if not occupied[y-s:y+s, x-s:x+s].any():
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 0))
                occupied[y-s:y+s, x-s:x+s] = True
                count += 1

    border_thickness = 15
    draw.rectangle([0, 0, w - 1, h - 1], outline=ImageColor.getrgb(bg_color), width=border_thickness)
    img.save("static/output.png")

    dwg = svgwrite.Drawing("static/output.svg", size=(w, h))
    occupied_svg = np.zeros((h, w), dtype=bool)
    count = 0
    for cnt in contours:
        for i, pt in enumerate(cnt[::2]):
            x, y = pt[0]
            dx, dy = x - w // 2, y - h // 2
            dist = np.sqrt(dx**2 + dy**2)
            norm = dist / np.sqrt((w // 2)**2 + (h // 2)**2)
            radius = int(1 + (1 - norm) * 3)
            buffer = radius + 1
            x1, y1, x2, y2 = x - buffer, y - buffer, x + buffer, y + buffer
            if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
                continue
            if not occupied_svg[y1:y2, x1:x2].any():
                if shape_type == "circle":
                    dwg.add(dwg.circle(center=(float(x), float(y)), r=radius, fill='black', stroke='none'))
                elif shape_type == "square":
                    dwg.add(dwg.rect(insert=(float(x-radius), float(y-radius)), size=(2*radius, 2*radius), fill='black'))
                elif shape_type == "triangle":
                    points = [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)]
                    dwg.add(dwg.polygon(points=[(float(px), float(py)) for px, py in points], fill='black'))
                elif shape_type == "sand":
                    path = f"M{x},{y+radius//2} C{x-radius},{y-radius} {x+radius},{y-radius} {x},{y+radius//2} Z"
                    dwg.add(dwg.path(d=path, fill='black'))
                elif shape_type == "realHeart":
                    path = f"M{x},{y} C{x - radius},{y - radius} {x - radius},{y - 2 * radius} {x},{y - radius} " + \
                           f"C{x + radius},{y - 2 * radius} {x + radius},{y - radius} {x},{y} Z"
                    dwg.add(dwg.path(d=path, fill='black'))
                elif shape_type == "S":
                    path = f"M{x - radius},{y - radius} A{radius},{radius} 0 0,1 {x + radius},{y} " + \
                           f"A{radius},{radius} 0 0,1 {x - radius},{y + radius}"
                    dwg.add(dwg.path(d=path, fill='black'))
                elif shape_type == "I":
                    dwg.add(dwg.rect(insert=(float(x - radius // 3), float(y - radius)), size=(float(2 * radius // 3), float(2 * radius)), fill='black'))
                occupied_svg[y1:y2, x1:x2] = True
                count += 1



    coords = np.argwhere(mask == 255)
    np.random.shuffle(coords)
    for y, x in coords:
        if count > 4000:
            break
        r = np.random.randint(1, 4)
        s = r + 1
        if 0 <= x-s < w and 0 <= y-s < h and x+s < w and y+s < h:
            if not occupied_svg[y-s:y+s, x-s:x+s].any():
                dwg.add(dwg.circle(center=(float(x), float(y)), r=r, fill='black', stroke='none'))
                occupied_svg[y-s:y+s, x-s:x+s] = True
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
        sofa_unterkante_y = int(bg_h * 0.5)
        pos_y = sofa_unterkante_y - new_height_px - 20

        thickness = 6
        draw_visible = ImageDraw.Draw(foreground_resized)
        draw_visible.rectangle([0, 0, foreground_resized.width - 1, foreground_resized.height - 1], outline=ImageColor.getrgb(bg_color), width=thickness)

        background.paste(foreground_resized, (pos_x, pos_y), foreground_resized)
        background.convert("RGB").save(preview_path)

    create_preview("static/output.png", "static/background.jpg", width_cm, "static/preview.png")

    return render_template("index.html", result=True,
                           width_cm=width_cm,
                           height_cm=height_cm,
                           price=price,
                           image_uploaded=True)



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
