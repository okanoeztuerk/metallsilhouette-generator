import os
import uuid
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, render_template, request, session, jsonify, url_for
from PIL import Image, ImageDraw, ImageColor, ImageFilter
import svgwrite

# === 1) Pfade & Flask-Setup ===
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)
app.secret_key = "supersecretkey"

# === 2) U²-Net laden für KI-Segmentierung ===
u2net = torch.hub.load('NathanUA/U-2-Net', 'u2net', pretrained=True)
u2net.eval()
to_tensor = T.Compose([
    T.ToTensor(),
    T.Resize((320, 320)),
])

def segment_foreground(img_bgr: np.ndarray) -> np.ndarray:
    """
    Erzeugt eine 0/255-Maske für Personen/Tiere mit U²-Net.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp     = to_tensor(img_rgb).unsqueeze(0)  # [1,3,320,320]
    with torch.no_grad():
        pred = u2net(inp)[0][0].cpu().numpy()  # [H,W] in [0,1]
    mask_small = cv2.resize(pred, (img_bgr.shape[1], img_bgr.shape[0]))
    mask_bin   = (mask_small > 0.5).astype(np.uint8) * 255
    # Morphology-Glätten
    kernel = np.ones((5,5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask_bin

def create_preview(fg_path, bg_path, width_cm_real, preview_path, color):
    """
    Legt output.png über background.jpg und erzeugt preview.png.
    """
    background = Image.open(bg_path).convert("RGBA")
    foreground = Image.open(fg_path).convert("RGBA")
    bg_w, bg_h = background.size

    wand_pixel = int(bg_w * 0.75)
    ppcm       = wand_pixel / 200.0
    new_w      = int(width_cm_real * ppcm)
    new_h      = int(new_w * foreground.height / foreground.width)

    fg_res = foreground.resize((new_w, new_h), Image.LANCZOS)
    pos_x  = (bg_w - new_w)//2
    pos_y  = int(bg_h*0.5) - new_h - 20

    draw_v = ImageDraw.Draw(fg_res)
    draw_v.rectangle(
        [0, 0, new_w-1, new_h-1],
        outline=ImageColor.getrgb(color),
        width=6
    )

    background.paste(fg_res, (pos_x, pos_y), fg_res)
    background.convert("RGB").save(preview_path)

def process_image(input_path: str, base_dir: str, width_cm: float, color: str, shape_type: str) -> dict:
    """
    1) KI-Segmentierung
    2) Punkt-/Form-Pipeline auf Maske
    3) Speichern output.png + output.svg
    4) Vorschau preview.png
    """
    # --- 1) Maske ---
    img_bgr = cv2.imread(input_path)
    mask    = segment_foreground(img_bgr)
    h, w    = mask.shape
    coords  = np.argwhere(mask == 255)
    np.random.shuffle(coords)

    # --- 2) PNG mit Formen ---
    img      = Image.new("RGBA", (w, h), (*ImageColor.getrgb(color), 255))
    draw     = ImageDraw.Draw(img)
    occupied = np.zeros((h, w), dtype=bool)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        for pt in cnt[::2]:
            x, y = pt[0]
            dx, dy = x - w//2, y - h//2
            dist = np.hypot(dx, dy)
            norm = dist / np.hypot(w//2, h//2)
            r = int(1 + (1 - norm) * 3)
            s = r + 2
            x1, y1, x2, y2 = x-s, y-s, x+s, y+s

            if 0 <= x1 < w and 0 <= y1 < h and x2 < w and y2 < h:
                if not occupied[y1:y2, x1:x2].any():
                    # alle shape-Typen
                    if shape_type == "circle":
                        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,255,255,0))
                    elif shape_type == "square":
                        draw.rectangle((x-r, y-r, x+r, y+r), fill=(255,255,255,0))
                    elif shape_type == "triangle":
                        pts = [(x, y-r), (x-r, y+r), (x+r, y+r)]
                        draw.polygon(pts, fill=(255,255,255,0))
                    elif shape_type == "realHeart":
                        heart = Image.new("L", (2*r, 2*r), 0)
                        hd    = ImageDraw.Draw(heart)
                        hd.pieslice([0,0,2*r,2*r], 180,360, fill=255)
                        hd.polygon([(0,r),(r*2,r),(r,2*r)], fill=255)
                        img.paste(Image.new("RGBA", heart.size, (255,255,255,0)),
                                  (x-r, y-r), heart)
                    elif shape_type == "S":
                        s_path = Image.new("L", (2*r, 3*r), 0)
                        sd     = ImageDraw.Draw(s_path)
                        sd.arc([0,0,2*r,2*r], 0,180, fill=255)
                        sd.arc([0,r,2*r,3*r], 180,360, fill=255)
                        img.paste(Image.new("RGBA", s_path.size, (255,255,255,0)),
                                  (x-r, y-r), s_path)
                    elif shape_type == "I":
                        draw.rectangle((x-r//3, y-r, x+r//3, y+r), fill=(255,255,255,0))

                    occupied[y1:y2, x1:x2] = True

    # Zufallspunkte
    count = 0
    for y, x in coords:
        if count > 2000: break
        r = np.random.randint(1, 4)
        s = r + 1
        if 0 <= x-s < w and 0 <= y-s < h and x+s < w and y+s < h:
            if not occupied[y-s:y+s, x-s:x+s].any():
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,255,255,0))
                occupied[y-s:y+s, x-s:x+s] = True
                count += 1

    # Rahmen
    draw.rectangle([0,0,w-1,h-1], outline=ImageColor.getrgb(color), width=15)

    # output.png
    output_png = os.path.join(base_dir, "output.png")
    img.save(output_png)

    # --- 3) SVG ---
    dwg = svgwrite.Drawing(os.path.join(base_dir, "output.svg"), size=(w, h))
    occupied_svg = np.zeros((h, w), dtype=bool)
    for cnt in contours:
        for pt in cnt[::2]:
            x, y = pt[0]
            dx, dy = x - w//2, y - h//2
            dist = np.hypot(dx, dy)
            norm = dist / np.hypot(w//2, h//2)
            radius = int(1 + (1 - norm) * 3)
            buffer = radius + 1
            x1, y1, x2, y2 = x-buffer, y-buffer, x+buffer, y+buffer
            if 0 <= x1 < w and 0 <= y1 < h and x2 < w and y2 < h:
                if not occupied_svg[y1:y2, x1:x2].any():
                    dwg.add(dwg.circle(
                        cx=float(x), cy=float(y), r=float(radius),
                        fill='black', stroke='none'
                    ))
                    occupied_svg[y1:y2, x1:x2] = True
    dwg.save()

    # --- 4) Preview ---
    preview_png = os.path.join(base_dir, "preview.png")
    create_preview(
        output_png,
        os.path.join(app.static_folder, "background.jpg"),
        width_cm,
        preview_png,
        color
    )

    return {"preview": preview_png, "png": output_png, "svg": dwg.filename}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=False)


@app.route("/widget", methods=["GET"])
def widget():
    return render_template("widget.html")


@app.route("/api/generate-shopify", methods=["POST"])
def generate_shopify():
    try:
        # 1) Felder lesen
        file        = request.files.get("image")
        fmt         = request.form["format"]
        orientation = request.form.get("orientation", "portrait")
        color       = request.form["color"]
        shape_type  = request.form["shape"]

        if not file:
            return jsonify({"error": "Kein Bild hochgeladen"}), 400

        # 2) Format → cm
        fmt_map = {"50x70":(50,70), "70x100":(70,100), "100x140":(100,140)}
        if fmt not in fmt_map:
            return jsonify({"error": f"Unbekanntes Format {fmt}"}), 400
        w_cm, h_cm = fmt_map[fmt]
        if orientation == "landscape":
            w_cm, h_cm = h_cm, w_cm

        # 3) Ordnerstruktur
        image_uid = str(uuid.uuid4())
        generated_parent = os.path.join(app.static_folder, "generated")
        os.makedirs(generated_parent, exist_ok=True)
        base_dir = os.path.join(generated_parent, image_uid)
        os.makedirs(base_dir, exist_ok=True)

        # 4) Input speichern
        input_path = os.path.join(base_dir, "input.jpg")
        file.save(input_path)

        # 5) Center-Crop
        img = Image.open(input_path)
        orig_w, orig_h = img.size
        tr = h_cm / w_cm
        or_ = orig_h / orig_w
        if or_ > tr:
            new_h = int(tr * orig_w)
            top   = (orig_h - new_h)//2
            img = img.crop((0, top, orig_w, top + new_h))
        else:
            new_w = int(orig_h / tr)
            left  = (orig_w - new_w)//2
            img = img.crop((left, 0, left + new_w, orig_h))
        img.save(input_path)

        # 6) Prozessieren
        paths = process_image(input_path, base_dir, w_cm, color, shape_type)

        # 7) Antwort
        return jsonify({
            "type": "wandbild_ready",
            "output_preview_url": url_for('static', filename=f"generated/{image_uid}/preview.png", _external=True),
            "output_png_url":     url_for('static', filename=f"generated/{image_uid}/output.png",  _external=True),
            "output_svg_url":     url_for('static', filename=f"generated/{image_uid}/output.svg",  _external=True),
            "image_uid": image_uid
        })

    except Exception as e:
        app.logger.exception("Fehler in /api/generate-shopify")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


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

