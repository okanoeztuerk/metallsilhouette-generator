import os
import cv2
import numpy as np
from flask import Flask, render_template, request, session, jsonify, abort
from PIL import Image, ImageDraw, ImageColor, ImageFilter
import svgwrite
import uuid
import hmac, hashlib

app = Flask(__name__)
app.secret_key = "supersecretkey"
SHOPIFY_WEBHOOK_SECRET = os.environ.get("SHOPIFY_WEBHOOK_SECRET", "")





def verify_hmac(data, hmac_header):
    hm = hmac.new(SHOPIFY_WEBHOOK_SECRET.encode(), data, hashlib.sha256)
    return hmac.compare_digest(hm.hexdigest(), hmac_header)

@app.route('/api/webhook/orders/create', methods=['POST'])
def orders_create_webhook():
    # 1) HMAC prüfen
    hmac_header = request.headers.get('X-Shopify-Hmac-Sha256', '')
    raw = request.get_data()
    if not verify_hmac(raw, hmac_header):
        return abort(401)

    # 2) Payload parsen
    order = request.get_json()
    order_id = order['id']

    # 3) Zeilen durchgehen
    for item in order.get('line_items', []):
        uid = None
        for k,v in item.get('properties', []):
            if k == 'ImageUID':
                uid = v
        if uid:
            # 4) Dateien verschieben oder in DB speichern
            src = os.path.join('static', 'generated', uid)
            dest = os.path.join('static', 'orders', str(order_id), uid)
            os.makedirs(dest, exist_ok=True)
            for fn in ['input.jpg','preview.png','output.png','output.svg']:
                shutil.move(os.path.join(src, fn), os.path.join(dest, fn))
    return '', 200

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=False, image_uploaded=os.path.exists("static/upload.jpg"))


@app.route("/widget")
def widget():
    return render_template("widget.html")

def process_image(input_path: str, base_dir: str, width_cm: float, color: str, shape_type: str):
    """
    Lädt input_path, wendet deine k-means Segmentierung,
    Punkt-/Form-Zeichnung und SVG-Ausgabe an und speichert:
      - preview.png  (für Vorschau)
      - output.png   (reines Wandbild-PNG)
      - output.svg   (reines Wandbild-SVG)
    alles unter base_dir.
    """

    # === 1) Bild laden & segmentieren ===
    image_bgr = cv2.imread(input_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = image_rgb.shape[:2]
    image_resized = cv2.resize(image_rgb, (512, int(h0 * 512 / w0)))
    lab = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = labels.flatten().reshape(image_resized.shape[:2])

    mask = np.zeros_like(segmented, dtype=np.uint8)
    for i, c in enumerate(centers):
        l,a,b = c
        if 100 < l < 200 and 115 < a < 145 and 115 < b < 145:
            mask[segmented == i] = 255

    h, w = mask.shape
    coords = np.argwhere(mask == 255)
    np.random.shuffle(coords)

    # === 2) PNG mit Punkten/Formen zeichnen ===
    img = Image.new("RGBA", (w,h), (*ImageColor.getrgb(color),255))
    draw = ImageDraw.Draw(img)
    occupied = np.zeros((h,w), dtype=bool)

    # Konturen-Punktwolken
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # Fülle weitere zufällige Punkte
    count=0
    for y,x in coords:
        if count>2000: break
        r = np.random.randint(1,4)
        s = r+1
        if 0<=x-s and 0<=y-s and x+s<w and y+s<h and not occupied[y-s:y+s,x-s:x+s].any():
            draw.ellipse((x-r,y-r,x+r,y+r), fill=(255,255,255,0))
            occupied[y-s:y+s,x-s:x+s] = True
            count+=1

    # Rahmen
    border_thickness = 15
    draw.rectangle([0, 0, w - 1, h - 1], outline=ImageColor.getrgb(color), width=border_thickness)
    img.save("static/output.png")


    # === SVG erzeugen ===
    dwg = svgwrite.Drawing(os.path.join(base_dir, "output.svg"), size=(w, h))
    occupied_svg = np.zeros((h, w), dtype=bool)
    count = 0

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

    dwg.save()


    # === 4) Vorschau mit Hintergrund erzeugen ===
    def create_preview2(fg_path, bg_path, width_cm_real, preview_path):
        background = Image.open(bg_path).convert("RGBA")
        foreground = Image.open(fg_path).convert("RGBA")
        bg_w, bg_h = background.size
        wand_pixel = int(bg_w * 0.75)
        ppcm = wand_pixel / 200.0
        new_w = int(width_cm_real * ppcm)
        new_h = int(new_w * foreground.height / foreground.width)
        fg_res = foreground.resize((new_w,new_h), Image.LANCZOS)
        pos_x = (bg_w-new_w)//2
        pos_y = int(bg_h*0.5) - new_h - 20
        draw_v = ImageDraw.Draw(fg_res)
        draw_v.rectangle([0,0,new_w-1,new_h-1], outline=ImageColor.getrgb(color), width=6)
        background.paste(fg_res, (pos_x,pos_y), fg_res)
        background.convert("RGB").save(preview_path)

    preview_png = os.path.join(base_dir, "preview.png")
    create_preview2(output_png, "static/background.jpg", width_cm, preview_png)

    return {
        "preview": preview_png,
        "png":      output_png,
        "svg":      os.path.join(base_dir, "output.svg")
    }

@app.route("/api/generate-shopify", methods=["POST"])
def generate_shopify():
    try:
        # 1) Parameter auslesen
        width_cm   = float(request.form["width_cm"])
        color      = request.form["color"]
        shape_type = request.form["shape"]

        # 2) Neue UUID und Verzeichnis anlegen
        image_uid = str(uuid.uuid4())
        base_dir = os.path.join("static", "generated", image_uid)
        os.makedirs(base_dir, exist_ok=True)

        # 3) Eingabebild speichern
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Kein Bild hochgeladen"}), 400
        input_path = os.path.join(base_dir, "input.jpg")
        file.save(input_path)

        # 4) Prüfe, ob background.jpg existiert
        bg_path = os.path.join("static", "background.jpg")
        if not os.path.isfile(bg_path):
            return jsonify({"error": "Hintergrundbild static/background.jpg fehlt"}), 500

        # 5) Bildverarbeitung (PNG, SVG, Vorschau) aufrufen
        paths = process_image(input_path, base_dir, width_cm, color, shape_type)

        # 6) Absolute URLs zusammensetzen und zurückgeben
        base_url = request.url_root.rstrip("/")
        return jsonify({
            "type": "wandbild_ready",
            "output_preview_url": f"{base_url}/{paths['preview']}",
            "output_png_url":     f"{base_url}/{paths['png']}",
            "output_svg_url":     f"{base_url}/{paths['svg']}",
            "image_uid":          image_uid
        })

    except Exception as e:
        # Logge den kompletten Trace ins Server-Log
        app.logger.exception("Fehler in /api/generate-shopify")
        # Gib die Fehlermeldung im JSON zurück, damit du sie in widget.html siehst
        return jsonify({"error": str(e)}), 500



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
