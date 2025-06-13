from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
from werkzeug.utils import secure_filename
import io
import os
import numpy as np
import cv2
import base64
import fitz  # PyMuPDF for PDF/image conversion
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, TimeoutException
import time
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from flask import send_file
import io
from flask import current_app
import json
import torch
from diffusers import StableDiffusionPipeline
import zipfile
from flask import make_response
from reportlab.lib.utils import ImageReader
import qrcode
import qrcode
import base64
from PIL import Image
from flask import render_template, request

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Prepare rembg sessions for different models
rembg_session_fast = new_session("u2netp")  # Fast, less accurate
rembg_session_high = new_session("u2net")   # High quality, more accurate

@app.route('/watermark', methods=['GET', 'POST'])
def watermark():
    result_url = None
    error = None
    positions = {
        'bottom-right': lambda img, w, h, margin: (img.width - w - margin, img.height - h - margin),
        'bottom-left': lambda img, w, h, margin: (margin, img.height - h - margin),
        'top-right': lambda img, w, h, margin: (img.width - w - margin, margin),
        'top-left': lambda img, w, h, margin: (margin, margin),
        'center': lambda img, w, h, margin: ((img.width - w) // 2, (img.height - h) // 2)
    }
    if request.method == 'POST':
        image_file = request.files.get('image')
        watermark_text = request.form.get('watermark_text', '').strip()
        watermark_icon = request.files.get('watermark_icon')
        position_key = request.form.get('watermark_position', 'bottom-right')
        if not image_file:
            error = "Please upload an image."
        else:
            try:
                # Save uploaded image
                img = Image.open(image_file.stream).convert("RGBA")
                img_filename = secure_filename(image_file.filename)
                base_name, ext = os.path.splitext(img_filename)
                output_filename = f"{base_name}_watermarked.png"
                output_path = os.path.join('static', 'results', output_filename)

                # Create watermark layer
                watermark_layer = Image.new("RGBA", img.size, (0,0,0,0))
                margin = 20

                if watermark_icon and watermark_icon.filename:
                    # Use icon/logo as watermark
                    icon = Image.open(watermark_icon.stream).convert("RGBA")
                    # Resize icon to fit (e.g., 20% of image width)
                    icon_width = int(img.width * 0.2)
                    icon_ratio = icon_width / icon.width
                    icon_height = int(icon.height * icon_ratio)
                    icon = icon.resize((icon_width, icon_height), Image.LANCZOS)
                    # Position: bottom-right with margin
                    pos = positions.get(position_key, positions['bottom-right'])(img, icon_width, icon_height, margin)
                    # Paste icon onto watermark layer with transparency
                    watermark_layer.paste(icon, pos, icon)
                elif watermark_text:
                    # Use text as watermark
                    draw = ImageDraw.Draw(watermark_layer)
                    font_size = int(img.width * 0.045)
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    text = watermark_text
                    text_width, text_height = draw.textsize(text, font=font)
                    pos = positions.get(position_key, positions['bottom-right'])(img, text_width, text_height, margin)
                    draw.text(pos, text, font=font, fill=(255,255,255,180))
                else:
                    error = "Please provide watermark text or icon/logo."

                if not error:
                    # Composite watermark layer onto image
                    watermarked = Image.alpha_composite(img, watermark_layer)
                    # Save result
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    watermarked.convert("RGB").save(output_path, "PNG")
                    result_url = url_for('static', filename=f"results/{output_filename}")

            except Exception as e:
                error = f"Error processing image: {e}"

    return render_template('watermark.html', result_url=result_url, error=error)

@app.route('/compress', methods=['GET', 'POST'])
def compress():
    result_url = None
    error = None
    if request.method == 'POST':
        file = request.files.get('image')
        quality = int(request.form.get('quality', 80))
        if not file or file.filename == '':
            error = 'No image uploaded.'
        else:
            img = Image.open(file)
            # Convert RGBA to RGB if needed for JPEG
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            out_path = os.path.join(RESULT_FOLDER, f'compressed_{file.filename}')
            img.save(out_path, quality=quality, optimize=True)
            result_url = url_for('show_result', filename=f'compressed_{file.filename}')
    return render_template('compress.html', result_url=result_url, error=error)

@app.route('/object_eraser', methods=['GET', 'POST'])
def object_eraser():
    result_url = None
    error = None
    if request.method == 'POST':
        file = request.files.get('image')
        mask_file = request.files.get('mask')
        if not file or file.filename == '' or not mask_file or mask_file.filename == '':
            error = 'Upload both image and mask.'
        else:
            img = Image.open(file).convert('RGB')
            mask = Image.open(mask_file).convert('L')
            pipe = get_sd_pipe()
            result = pipe(image=img, mask_image=mask, prompt="inpaint", guidance_scale=7.5).images[0]
            out_path = os.path.join(RESULT_FOLDER, f'inpainted_{file.filename}')
            result.save(out_path)
            result_url = url_for('show_result', filename=f'inpainted_{file.filename}')
    return render_template('object_eraser.html', result_url=result_url, error=error)

@app.route('/crop_resize', methods=['GET', 'POST'])
def crop_resize():
    result_url = None
    error = None
    if request.method == 'POST':
        file = request.files.get('image')
        x = int(request.form.get('x', 0))
        y = int(request.form.get('y', 0))
        w = int(request.form.get('w', 100))
        h = int(request.form.get('h', 100))
        resize_w = int(request.form.get('resize_w', w))
        resize_h = int(request.form.get('resize_h', h))
        if not file or file.filename == '':
            error = 'No image uploaded.'
        else:
            img = Image.open(file)
            cropped = img.crop((x, y, x + w, y + h))
            resized = cropped.resize((resize_w, resize_h), Image.LANCZOS)
            out_path = os.path.join(RESULT_FOLDER, f'cropped_{file.filename}')
            resized.save(out_path)
            result_url = url_for('show_result', filename=f'cropped_{file.filename}')
    return render_template('crop_resize.html', result_url=result_url, error=error)

@app.route('/batch_bg_remover', methods=['GET', 'POST'])
def batch_bg_remover():
    zip_url = None
    error = None
    if request.method == 'POST':
        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            error = 'No images uploaded.'
            return render_template('batch_bg_remover.html', error=error)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            for file in files:
                try:
                    img = Image.open(file)
                    output = remove(img)
                    out_name = f'bg_removed_{file.filename}'
                    out_bytes = io.BytesIO()
                    output.save(out_bytes, format='PNG')
                    out_bytes.seek(0)
                    zipf.writestr(out_name, out_bytes.read())
                except Exception as e:
                    continue
        zip_buffer.seek(0)
        zip_b64 = base64.b64encode(zip_buffer.read()).decode('utf-8')
        zip_url = f"data:application/zip;base64,{zip_b64}"
    return render_template('batch_bg_remover.html', zip_url=zip_url, error=error)
@app.route('/qr_generator', methods=['GET', 'POST'])
def qr_generator():
    qr_code_url = None
    if request.method == 'POST':
        qr_text = request.form.get('qr_text', '')
        fg_color = request.form.get('fg_color', '#000000')
        bg_color = request.form.get('bg_color', '#ffffff')
        icon_file = request.files.get('icon')

        if qr_text:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(qr_text)
            qr.make(fit=True)
            img: PilImage = qr.make_image(fill_color=fg_color, back_color=bg_color).convert('RGBA')

            # If icon is uploaded, paste it at the center
            if icon_file and icon_file.filename:
                icon = Image.open(icon_file).convert("RGBA")
                # Resize icon
                icon_size = min(img.size[0] // 4, 80)
                icon = icon.resize((icon_size, icon_size), Image.LANCZOS)
                # Calculate position
                pos = ((img.size[0] - icon_size) // 2, (img.size[1] - icon_size) // 2)
                img.paste(icon, pos, mask=icon)

            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            qr_code_url = f"data:image/png;base64,{img_b64}"
    return render_template('qr_generator.html', qr_code_url=qr_code_url)


# Load Stable Diffusion pipeline (v1.5 or similar) on startup
sd_pipe = None
sd_device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_sd_pipe():
    global sd_pipe
    if sd_pipe is None:
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if sd_device=='cuda' else torch.float32
        )
        sd_pipe = sd_pipe.to(sd_device)
    return sd_pipe

@app.route('/show_upload/<filename>')
def show_upload(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path)

@app.route('/remove_bg_action/<filename>')
def remove_bg_action(filename):
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    with Image.open(input_path) as img:
        output = remove(img)
        output_path = os.path.join(RESULT_FOLDER, f'result_{filename}')
        output.save(output_path, format='PNG')
    return redirect(url_for('result', filename=f'result_{filename}'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)
            # Copy to results/ so all features work directly after upload
            result_path = os.path.join(RESULT_FOLDER, file.filename)
            if not os.path.exists(result_path):
                import shutil
                shutil.copyfile(input_path, result_path)
            return redirect(url_for('choose_action', filename=file.filename))
    return render_template('index.html')

@app.route('/choose_action/<filename>')
def choose_action(filename):
    return render_template('choose_action.html', filename=filename)

@app.route('/result/<filename>')
def result(filename):
    import os
    path = os.path.join('results', filename)
    if not os.path.exists(path):
        return render_template('result.html', filename=filename, error='Result image not found. Please try again.')
    # Pass a URL to a dedicated route for serving result images
    image_url = url_for('show_result', filename=filename)
    return render_template('result.html', filename=filename, image_url=image_url)

@app.route('/show_result/<filename>')
def show_result(filename):
    path = os.path.join('results', filename)
    if not os.path.exists(path):
        from flask import abort
        abort(404)
    return send_file(path)

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    return send_file(path, as_attachment=True)

@app.route('/show/<filename>')
def show(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    return send_file(path)

@app.route('/edit/<filename>', methods=['GET', 'POST'])
def edit(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    edited_path = os.path.join(RESULT_FOLDER, f'edited_{filename}')
    if not os.path.exists(path):
        return render_template('edit.html', filename=filename, error='Image not found. Please process the image first.')
    if request.method == 'POST':
        action = request.form.get('action')
        value = float(request.form.get('value', 1))
        with Image.open(path) as img:
            if action == 'rotate':
                img = img.rotate(value, expand=True)
            elif action == 'flip_horizontal':
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif action == 'flip_vertical':
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif action == 'brightness':
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(value)
            elif action == 'contrast':
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(value)
            img.save(edited_path, format='PNG')
        return redirect(url_for('result', filename=f'edited_{filename}'))
    return render_template('edit.html', filename=filename)

@app.route('/replace_bg/<filename>', methods=['GET', 'POST'])
def replace_bg(filename):
    fg_path = os.path.join(RESULT_FOLDER, filename)
    if request.method == 'POST':
        bg_file = request.files.get('background')
        color = request.form.get('color')
        with Image.open(fg_path) as fg:
            fg = fg.convert('RGBA')
            if bg_file and bg_file.filename:
                bg_path = os.path.join(UPLOAD_FOLDER, 'bg_' + bg_file.filename)
                bg_file.save(bg_path)
                with Image.open(bg_path) as bg:
                    bg = bg.convert('RGBA').resize(fg.size)
                    out = Image.alpha_composite(bg, fg)
            elif color:
                bg = Image.new('RGBA', fg.size, color)
                out = Image.alpha_composite(bg, fg)
            else:
                out = fg
            out_path = os.path.join(RESULT_FOLDER, f'bg_{filename}')
            out.save(out_path, format='PNG')
        return redirect(url_for('result', filename=f'bg_{filename}'))
    return render_template('replace_bg.html', filename=filename)

@app.route('/upscale', methods=['GET', 'POST'])
def upscale_upload():
    before_image_url = None
    after_image_url = None
    filename = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('upscale.html', error='No image uploaded.')
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        # Copy to results for consistency
        result_path = os.path.join(RESULT_FOLDER, filename)
        Image.open(upload_path).save(result_path)
        # Redirect to /upscale/<filename> for consistent workflow
        return redirect(url_for('upscale', filename=filename))
    return render_template('upscale.html')

@app.route('/upscale/<filename>', methods=['GET', 'POST'])
def upscale(filename):
    before_image_url = url_for('show_result', filename=filename)
    after_image_url = None
    if request.method == 'POST':
        scale = int(request.form.get('scale', 2))
        path = os.path.join(RESULT_FOLDER, filename)
        upscaled_path = os.path.join(RESULT_FOLDER, f'upscaled_{scale}x_{filename}')
        with Image.open(path) as img:
            new_size = (img.width * scale, img.height * scale)
            upscaled = img.resize(new_size, Image.LANCZOS)
            upscaled.save(upscaled_path, format='PNG')
        after_image_url = url_for('show_result', filename=f'upscaled_{scale}x_{filename}')
        return render_template('upscale.html', before_image_url=before_image_url, after_image_url=after_image_url, filename=filename)
    return render_template('upscale.html', before_image_url=before_image_url, filename=filename)

@app.route('/manage', methods=['GET', 'POST'])
def manage():
    # List images from uploads and results
    upload_dir = UPLOAD_FOLDER
    result_dir = RESULT_FOLDER
    images = []
    for folder, url_prefix in [(upload_dir, 'uploads'), (result_dir, 'results')]:
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                images.append({
                    'name': fname,
                    'path': os.path.join(folder, fname),
                    'url': url_for('static', filename=f'{url_prefix}/{fname}') if url_prefix == 'uploads' else url_for('show', filename=fname)
                })
    if request.method == 'POST':
        to_delete = request.form.getlist('delete_imgs')
        for path in to_delete:
            try:
                os.remove(path)
            except Exception:
                pass
        return redirect(url_for('manage'))
    return render_template('manage.html', images=images)

@app.route('/auto_enhance/<filename>', methods=['GET', 'POST'])
def auto_enhance(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    enhanced_path = os.path.join(RESULT_FOLDER, f'autoenhanced_{filename}')
    if not os.path.exists(path):
        return render_template('result.html', filename=filename, error='Image not found.')
    if request.method == 'POST':
        with Image.open(path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img = ImageOps.autocontrast(img)
            img = ImageEnhance.Brightness(img).enhance(1.08)
            img = ImageEnhance.Contrast(img).enhance(1.12)
            img.save(enhanced_path, format='PNG')
        return redirect(url_for('result', filename=f'autoenhanced_{filename}'))
    return render_template('auto_enhance.html', filename=filename)

@app.route('/profile_pic_maker/<filename>', methods=['GET', 'POST'])
def profile_pic_maker(filename):
    before_image_url = None
    after_image_url = None
    filename = None
    error = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            error = 'No image uploaded.'
        else:
            filename = file.filename
            upload_path = os.path.join('results', filename)
            file.save(upload_path)
            # Open and process image (crop to square, resize, etc.)
            from PIL import Image
            img = Image.open(upload_path)
            min_side = min(img.width, img.height)
            left = (img.width - min_side) // 2
            top = (img.height - min_side) // 2
            img = img.crop((left, top, left + min_side, top + min_side))
            img = img.resize((400, 400), Image.LANCZOS)
            out_filename = f'profile_{filename}'
            out_path = os.path.join('results', out_filename)
            img.save(out_path)
            before_image_url = url_for('show_result', filename=filename)
            after_image_url = url_for('show_result', filename=out_filename)
    return render_template('profile_pic_maker.html', before_image_url=before_image_url, after_image_url=after_image_url, error=error)

@app.route('/smart_resize', methods=['GET', 'POST'])
def smart_resize():
    if request.method == 'POST':
        file = request.files.get('image')
        platform = request.form.get('platform')
        if not file or file.filename == '':
            return render_template('smart_resize.html', error='No image uploaded.')
        img = Image.open(file)
        # Example: Resize for Instagram (1080x1080), LinkedIn (400x400), YouTube (1280x720)
        sizes = {'instagram': (1080, 1080), 'linkedin': (400, 400), 'youtube': (1280, 720)}
        size = sizes.get(platform, (1080, 1080))
        img = ImageOps.fit(img, size, Image.LANCZOS)
        out_path = os.path.join('results', f'smartresize_{file.filename}')
        img.save(out_path)
        return render_template('result.html', filename=f'smartresize_{file.filename}')
    return render_template('smart_resize.html')

@app.route('/resume_shot', methods=['GET', 'POST'])
def resume_shot():
    if request.method == 'POST':
        file = request.files.get('image')
        template = request.form.get('template', 'classic')
        if not file or file.filename == '':
            return render_template('resume_shot.html', error='No image uploaded.')
        img = Image.open(file)
        img = ImageOps.fit(img, (300, 400), Image.LANCZOS)
        out_img_name = f'resume_{file.filename}'
        out_img_path = os.path.join('results', out_img_name)
        img.save(out_img_path)
        # Generate PDF with selected template
        out_pdf_name = f'resume_{os.path.splitext(file.filename)[0]}_{template}.pdf'
        out_pdf_path = os.path.join('results', out_pdf_name)
        c = canvas.Canvas(out_pdf_path, pagesize=A4)
        width, height = A4
        # Draw gradient background based on template
        if template == 'classic':
            c.setFillColor(HexColor('#f8f8ff'))
            c.rect(0, 0, width, height, fill=1, stroke=0)
        elif template == 'purple':
            c.setFillColor(HexColor('#8F42FF'))
            c.rect(0, 0, width, height, fill=1, stroke=0)
            c.setFillColor(HexColor('#fff'))
            c.roundRect(40, 80, width-80, height-160, 30, fill=1, stroke=0)
        elif template == 'blue-gradient':
            # Simulate a blue gradient with rectangles
            for i in range(0, int(height), 10):
                c.setFillColor(HexColor(f'#%02x%02x%02x' % (120, 180, 255 - int(i/height*100))))
                c.rect(0, i, width, 10, fill=1, stroke=0)
        # Place image in center
        img_reader = ImageReader(out_img_path)
        img_w, img_h = img.size
        c.drawImage(img_reader, (width-img_w)/2, height/2-100, img_w, img_h)
        # Add name/title placeholder
        c.setFont('Helvetica-Bold', 22)
        c.setFillColor(HexColor('#6A0DAD'))
        c.drawCentredString(width/2, height/2+170, 'Your Name Here')
        c.setFont('Helvetica', 14)
        c.setFillColor(HexColor('#333'))
        c.drawCentredString(width/2, height/2+145, 'Your Profession / Title')
        c.save()
        before_url = url_for('show_upload', filename=file.filename)
        after_url = url_for('show_result', filename=out_img_name)
        pdf_url = url_for('show_result', filename=out_pdf_name)
        return render_template('resume_shot.html', before_image_url=before_url, after_image_url=after_url, pdf_url=pdf_url, template_selected=template)
    return render_template('resume_shot.html')

@app.route('/color_corrector', methods=['GET', 'POST'])
def color_corrector():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('color_corrector.html', error='No image uploaded.')
        img = Image.open(file)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Brightness(img).enhance(1.2)
        out_path = os.path.join('results', f'colorcorrect_{file.filename}')
        img.save(out_path)
        return render_template('result.html', filename=f'colorcorrect_{file.filename}')
    return render_template('color_corrector.html')

@app.route('/sketch_enhancer', methods=['GET', 'POST'])
def sketch_enhancer():
    if request.method == 'POST':
        # If this is the initial upload
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            before_url = url_for('show_upload', filename=filename)
            return render_template('sketch_enhancer.html', before_image_url=before_url, after_image_url=None, filename=filename)
        # If this is the sketch creation step
        elif request.form.get('sketch') == '1' and request.form.get('filename'):
            filename = request.form['filename']
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                # Read image as OpenCV array
                img = cv2.imread(upload_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 1. Use a bilateral filter for edge-preserving smoothing
                smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
                # 2. Invert and blur for dodge blend
                inv = 255 - smooth
                blur = cv2.GaussianBlur(inv, (25, 25), sigmaX=0, sigmaY=0)
                def dodgeV2(x, y):
                    return cv2.divide(x, 255 - y, scale=256)
                sketch = dodgeV2(smooth, blur)
                # 3. Optional: Slightly enhance contrast for more pencil-like look
                sketch = cv2.equalizeHist(sketch)
                out_filename = f'sketch_{filename}'
                out_path = os.path.join(RESULT_FOLDER, out_filename)
                cv2.imwrite(out_path, sketch)
                before_url = url_for('show_upload', filename=filename)
                after_url = url_for('show_result', filename=out_filename)
                return render_template('sketch_enhancer.html', before_image_url=before_url, after_image_url=after_url, filename=filename)
            except Exception as e:
                return render_template('sketch_enhancer.html', error=f'Processing failed: {e}')
        else:
            return render_template('sketch_enhancer.html', error='No image uploaded.')
    return render_template('sketch_enhancer.html')

@app.route('/note_cleaner', methods=['GET', 'POST'])
def note_cleaner():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('note_cleaner.html', error='No image uploaded.')
        img = Image.open(file).convert('L')
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        out_path = os.path.join('results', f'note_{file.filename}')
        img.save(out_path)
        return render_template('result.html', filename=f'note_{file.filename}')
    return render_template('note_cleaner.html')

@app.route('/transparent_icon', methods=['GET', 'POST'])
def transparent_icon():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('transparent_icon.html', error='No image uploaded.')
        img = Image.open(file).convert('RGBA')
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        # Always save as PNG to support transparency
        base_filename = os.path.splitext(file.filename)[0]
        out_path = os.path.join('results', f'transparent_{base_filename}.png')
        img.save(out_path, format='PNG')
        return render_template('transparent_icon.html', filename=f'transparent_{base_filename}.png')
    return render_template('transparent_icon.html')

@app.route('/bg_remover', methods=['GET', 'POST'])
def bg_remover():
    before_image_url = None
    after_image_url = None
    filename = None
    if request.method == 'POST':
        file = request.files.get('image')
        # Optional: allow user to select quality
        quality = request.form.get('quality', 'high')  # 'high' or 'fast'
        # Save original upload for preview
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        img = Image.open(file)
        img.save(upload_path)
        before_image_url = url_for('show_upload', filename=file.filename)
        # Remove background using rembg with selected model
        # Resize image for better accuracy (optional, e.g. max 1024px)
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        if quality == 'high':
            output = remove(img, session=rembg_session_high)
        else:
            output = remove(img, session=rembg_session_fast)
        out_filename = f'bg_result_{file.filename}'
        out_path = os.path.join(RESULT_FOLDER, out_filename)
        output.save(out_path, format='PNG')
        after_image_url = url_for('show_result', filename=out_filename)
        filename = out_filename
        return render_template('bg_remover.html', before_image_url=before_image_url, after_image_url=after_image_url, filename=out_filename)
    return render_template('bg_remover.html', before_image_url=None, after_image_url=None, filename=None)

@app.route('/ai_auto_enhance', methods=['GET', 'POST'])
def ai_auto_enhance():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('auto_enhance.html', error='No image uploaded.')
        # Save original upload
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_path)
        # Open and enhance
        img = Image.open(upload_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = ImageOps.autocontrast(img)
        enhanced_filename = f'autoenhanced_{file.filename}'
        out_path = os.path.join(RESULT_FOLDER, enhanced_filename)
        img.save(out_path)
        before_image_url = url_for('show_upload', filename=file.filename)
        after_image_url = url_for('show_result', filename=enhanced_filename)
        return render_template('auto_enhance.html', before_image_url=before_image_url, after_image_url=after_image_url)
    return render_template('auto_enhance.html')

@app.route('/ai_background_replacement', methods=['GET', 'POST'])
def ai_background_replacement():
    before_image_url = None
    after_image_url = None
    filename = None
    if request.method == 'POST':
        # Step 1: Remove background
        file = request.files.get('image')
        bg_file = request.files.get('background')
        color = request.form.get('color')
        filename_field = request.form.get('filename')
        if file and file.filename != '':
            # User uploaded a new image, remove background
            img = Image.open(file).convert('RGBA')
            removed = remove(img)
            out_filename = f'bg_removed_{file.filename}'
            out_path = os.path.join(RESULT_FOLDER, out_filename)
            removed.save(out_path, format='PNG')
            before_image_url = url_for('show_upload', filename=file.filename)
            after_image_url = url_for('show_result', filename=out_filename)
            filename = out_filename
            return render_template('replace_bg.html', before_image_url=before_image_url, after_image_url=after_image_url, filename=filename)
        elif filename_field:
            # Step 2: Add background to already-processed image
            fg_path = os.path.join(RESULT_FOLDER, filename_field)
            if not os.path.exists(fg_path):
                return render_template('replace_bg.html', error='No image found for background replacement.', before_image_url=None, after_image_url=None, filename=None)
            with Image.open(fg_path).convert('RGBA') as fg:
                if bg_file and bg_file.filename:
                    bg_path = os.path.join(UPLOAD_FOLDER, 'bg_' + bg_file.filename)
                    bg_file.save(bg_path)
                    with Image.open(bg_path).convert('RGBA') as bg:
                        bg = bg.resize(fg.size)
                        out = Image.alpha_composite(bg, fg)
                        out_filename = f'bg_replaced_{filename_field}'
                        out_path = os.path.join(RESULT_FOLDER, out_filename)
                        out.save(out_path, format='PNG')
                        after_image_url = url_for('show_result', filename=out_filename)
                        before_image_url = url_for('show_result', filename=filename_field)
                        filename = out_filename
                elif color:
                    bg = Image.new('RGBA', fg.size, color)
                    out = Image.alpha_composite(bg, fg)
                    out_filename = f'bg_colored_{filename_field}'
                    out_path = os.path.join(RESULT_FOLDER, out_filename)
                    out.save(out_path, format='PNG')
                    after_image_url = url_for('show_result', filename=out_filename)
                    before_image_url = url_for('show_result', filename=filename_field)
                    filename = out_filename
                else:
                    return render_template('replace_bg.html', error='No background or color provided.', before_image_url=None, after_image_url=None, filename=None)
                return render_template('replace_bg.html', before_image_url=before_image_url, after_image_url=after_image_url, filename=filename)
        else:
            return render_template('replace_bg.html', error='No image uploaded.', before_image_url=None, after_image_url=None, filename=None)
    return render_template('replace_bg.html', before_image_url=None, after_image_url=None, filename=None)

@app.route('/ai_profile_picture_maker', methods=['GET', 'POST'])
def ai_profile_picture_maker():
    before_image_url = None
    after_image_url = None
    filename = None
    error = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            error = 'No image uploaded.'
        else:
            filename = file.filename
            upload_path = os.path.join('results', filename)
            file.save(upload_path)
            # Open and process image (crop to square, resize, etc.)
            from PIL import Image
            img = Image.open(upload_path)
            min_side = min(img.width, img.height)
            left = (img.width - min_side) // 2
            top = (img.height - min_side) // 2
            img = img.crop((left, top, left + min_side, top + min_side))
            img = img.resize((400, 400), Image.LANCZOS)
            out_filename = f'profile_{filename}'
            out_path = os.path.join('results', out_filename)
            img.save(out_path)
            before_image_url = url_for('show_result', filename=filename)
            after_image_url = url_for('show_result', filename=out_filename)
    return render_template('profile_pic_maker.html', before_image_url=before_image_url, after_image_url=after_image_url, error=error)

@app.route('/convert', methods=['GET', 'POST'])
def convert():
    download_url = None
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        convert_to = request.form.get('convert_to')
        if not file or file.filename == '':
            error = 'No file uploaded.'
            return render_template('convert.html', error=error)
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        out_filename = None
        out_path = None
        try:
            if ext in ['.jpg', '.jpeg', '.png'] and convert_to == 'pdf':
                # Image to PDF
                with Image.open(input_path) as img:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    out_filename = f'{name}.pdf'
                    out_path = os.path.join(RESULT_FOLDER, out_filename)
                    img.save(out_path, 'PDF', resolution=100.0)
            elif ext == '.pdf' and convert_to in ['jpg', 'png']:
                # PDF to Image (first page)
                doc = fitz.open(input_path)
                page = doc.load_page(0)
                pix = page.get_pixmap()
                out_filename = f'{name}_page1.{convert_to}'
                out_path = os.path.join(RESULT_FOLDER, out_filename)
                pix.save(out_path)
            elif ext in ['.jpg', '.jpeg', '.png'] and convert_to in ['jpg', 'png'] and ext[1:] != convert_to:
                # Image to Image
                with Image.open(input_path) as img:
                    if convert_to == 'jpg' and img.mode == 'RGBA':
                        img = img.convert('RGB')
                    out_filename = f'{name}.{convert_to}'
                    out_path = os.path.join(RESULT_FOLDER, out_filename)
                    img.save(out_path, format=convert_to.upper())
            elif ext == '.pdf' and convert_to == 'pdf':
                # PDF to PDF (copy)
                out_filename = f'{name}_copy.pdf'
                out_path = os.path.join(RESULT_FOLDER, out_filename)
                import shutil
                shutil.copyfile(input_path, out_path)
            else:
                error = 'Unsupported conversion.'
        except Exception as e:
            error = f'Conversion failed: {e}'
        if out_path and os.path.exists(out_path):
            download_url = url_for('download', filename=out_filename)
        return render_template('convert.html', download_url=download_url, error=error)
    return render_template('convert.html')

@app.route('/resume-generator')
def resume_generator():
    return render_template('resume_generator.html')

@app.route('/resume-builder', methods=['GET'])
def resume_builder():
    """
    Route for the advanced Smart Resume Builder page.
    Renders the multi-step resume builder UI with template selection, live preview, and PDF download.
    """
    return render_template('resume_builder.html')

@app.route('/smart-resume', methods=['GET'])
def smart_resume_generator():
    """
    Route for the new Smart Resume Generator page.
    Renders the modern, multi-section, template-switching resume builder UI.
    """
    return render_template('smart_resume_generator.html')

@app.route('/api/linkedin_parse', methods=['POST'])
def linkedin_parse():
    data = request.get_json()
    url = data.get('url')
    if not url or 'linkedin.com/in/' not in url:
        return jsonify({'success': False, 'error': 'Invalid LinkedIn profile URL.'})
    # Set up headless Chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(20)
        driver.get(url)
        time.sleep(3)  # Wait for page to load
        # Accept cookies if present
        try:
            accept_btn = driver.find_element(By.XPATH, "//button[contains(., 'Accept')]" )
            accept_btn.click()
            time.sleep(1)
        except Exception:
            pass
        # Parse public profile fields
        result = {'success': True}
        try:
            result['fullName'] = driver.find_element(By.CSS_SELECTOR, 'h1').text.strip()
        except Exception:
            result['fullName'] = ''
        try:
            result['headline'] = driver.find_element(By.CSS_SELECTOR, 'div.text-body-medium.break-words').text.strip()
        except Exception:
            result['headline'] = ''
        try:
            about = driver.find_element(By.XPATH, "//section[contains(@class,'summary')]//span[starts-with(@class,'visually-hidden')]")
            result['summary'] = about.text.strip()
        except Exception:
            result['summary'] = ''
        # Experience
        result['work'] = []
        try:
            exp_sections = driver.find_elements(By.XPATH, "//section[contains(@id,'experience')]//li[contains(@class,'artdeco-list__item')]")
            for exp in exp_sections:
                try:
                    title = exp.find_element(By.CSS_SELECTOR, 'span[aria-hidden="true"]').text.strip()
                except Exception:
                    title = ''
                try:
                    company = exp.find_element(By.CSS_SELECTOR, 'span.t-14.t-normal').text.strip()
                except Exception:
                    company = ''
                try:
                    year = exp.find_element(By.CSS_SELECTOR, 'span.t-14.t-normal.t-black--light').text.strip()
                except Exception:
                    year = ''
                try:
                    desc = exp.find_element(By.CSS_SELECTOR, 'div.pv-entity__extra-details').text.strip()
                except Exception:
                    desc = ''
                result['work'].append({'title': title, 'company': company, 'year': year, 'desc': desc})
        except Exception:
            pass
        # Education
        result['education'] = []
        try:
            edu_sections = driver.find_elements(By.XPATH, "//section[contains(@id,'education')]//li[contains(@class,'artdeco-list__item')]")
            for edu in edu_sections:
                try:
                    degree = edu.find_element(By.CSS_SELECTOR, 'span[aria-hidden="true"]').text.strip()
                except Exception:
                    degree = ''
                try:
                    institute = edu.find_element(By.CSS_SELECTOR, 'span.t-14.t-normal').text.strip()
                except Exception:
                    institute = ''
                try:
                    year = edu.find_element(By.CSS_SELECTOR, 'span.t-14.t-normal.t-black--light').text.strip()
                except Exception:
                    year = ''
                result['education'].append({'degree': degree, 'institute': institute, 'year': year})
        except Exception:
            pass
        # Skills (publicly visible)
        result['skills'] = []
        try:
            skills_section = driver.find_elements(By.XPATH, "//span[contains(@class,'pv-skill-category-entity__name-text')]//span[1]")
            for s in skills_section:
                skill = s.text.strip()
                if skill:
                    result['skills'].append(skill)
        except Exception:
            pass
        # Languages (publicly visible)
        result['languages'] = []
        try:
            lang_section = driver.find_elements(By.XPATH, "//section[contains(@id,'languages')]//li")
            for l in lang_section:
                lang = l.text.strip()
                if lang:
                    result['languages'].append(lang)
        except Exception:
            pass
        # Certifications (publicly visible)
        result['certifications'] = []
        try:
            cert_section = driver.find_elements(By.XPATH, "//section[contains(@id,'certifications')]//li")
            for c in cert_section:
                cert = c.text.strip()
                if cert:
                    result['certifications'].append(cert)
        except Exception:
            pass
        # Achievements (not always public)
        result['achievements'] = []
        # Projects (not always public)
        result['projects'] = []
        driver.quit()
        return jsonify(result)
    except WebDriverException as e:
        return jsonify({'success': False, 'error': f'Selenium error: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})

@app.route('/edit', methods=['GET', 'POST'])
def edit_upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('edit_upload.html', error='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('edit_upload.html', error='No selected file')
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)
            # Copy to results/ so edit/<filename> works
            result_path = os.path.join(RESULT_FOLDER, filename)
            if not os.path.exists(result_path):
                import shutil
                shutil.copyfile(input_path, result_path)
            return redirect(url_for('edit', filename=filename))
    return render_template('edit_upload.html')

# AI-powered suggestion endpoint (OpenAI-compatible placeholder)
@app.route('/api/resume_suggest', methods=['POST'])
def resume_suggest():
    """
    Accepts JSON with 'section' and 'content'. Returns AI-powered suggestions for resume improvement.
    """
    data = request.get_json()
    section = data.get('section', '')
    content = data.get('content', '')
    # TODO: Integrate with OpenAI or other LLM. For now, return a placeholder suggestion.
    # Example: Use openai.Completion.create(...) if you have an API key configured.
    suggestion = f"[AI Suggestion for {section}]: Consider making your {section} more concise and impactful. Example: ..."
    return jsonify({'success': True, 'suggestion': suggestion})

# PDF generation endpoint using pdfkit (HTML to PDF)
@app.route('/api/resume_pdf', methods=['POST'])
def resume_pdf():
    """
    Accepts JSON with 'html' and 'filename'. Returns a generated PDF file using pdfkit and wkhtmltopdf.
    """
    import pdfkit
    data = request.get_json()
    html_content = data.get('html', '')
    filename = data.get('filename', 'resume.pdf')
    pdf_io = io.BytesIO()
    # Configure pdfkit to use wkhtmltopdf
    config = pdfkit.configuration()
    pdf_bytes = pdfkit.from_string(html_content, False, configuration=config)
    pdf_io.write(pdf_bytes)
    pdf_io.seek(0)
    return send_file(pdf_io, mimetype='application/pdf', as_attachment=True, download_name=filename)

@app.route('/portrait_sketch')
def portrait_sketch():
    return render_template('portrait_sketch.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400
    pipe = get_sd_pipe()
    with torch.autocast(sd_device) if sd_device=='cuda' else torch.no_grad():
        image = pipe(prompt, guidance_scale=7.5).images[0]
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({'image': img_b64})

@app.route('/sketch', methods=['POST'])
def sketch():
    file = request.files.get('image')
    mode = request.form.get('mode', 'sketch')  # 'sketch' or 'cartoon'
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400
    in_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_bytes, cv2.IMREAD_COLOR)
    if mode == 'cartoon':
        # Cartoon effect
        # 1. Apply bilateral filter repeatedly for smooth color
        color = img.copy()
        for _ in range(2):
            color = cv2.bilateralFilter(color, 9, 75, 75)
        # 2. Convert to grayscale and median blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        # 3. Detect and enhance edges
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        # 4. Convert edges to color
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # 5. Combine color and edges using bitwise_and
        cartoon = cv2.bitwise_and(color, edges_colored)
        out_img = cartoon
    else:
        # Pencil sketch
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        out_img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode('.png', out_img)
    img_b64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({'image': img_b64})

@app.route('/portfolio-generator', methods=['GET', 'POST'])
def portfolio_generator():
    if request.method == 'POST':
        form = request.form
        # --- Skills ---
        skills = []
        for i in range(1, 5):
            skill = form.get(f'skill{i}_name', '').strip()
            if skill:
                skills.append(skill)
        skills_str = ', '.join(skills)
        # --- Experience ---
        experience = []
        for i in range(1, 4):
            role = form.get(f'exp{i}_role', '').strip()
            years = form.get(f'exp{i}_years', '').strip()
            desc = form.get(f'exp{i}_desc', '').strip()
            if role or years or desc:
                # Only join non-empty fields
                line = ' | '.join([x for x in [role, years, desc] if x])
                experience.append(line)
        experience_str = '\n'.join(experience)
        # --- Education ---
        education = []
        for i in range(1, 4):
            school = form.get(f'edu{i}_school', '').strip()
            years = form.get(f'edu{i}_years', '').strip()
            desc = form.get(f'edu{i}_desc', '').strip()
            if school or years or desc:
                line = ' | '.join([x for x in [school, years, desc] if x])
                education.append(line)
        education_str = '\n'.join(education)
        # --- Projects ---
        projects = []
        for i in range(1, 10):
            title = form.get(f'project{i}_title', '').strip()
            cat = form.get(f'project{i}_cat', '').strip()
            link = form.get(f'project{i}_link', '').strip()
            if title or cat or link:
                line = ' | '.join([x for x in [title, cat, link] if x])
                projects.append(line)
        projects_str = '\n'.join(projects)
        # --- Links ---
        links = form.get('links', '').strip()
        # --- Main fields ---
        data = {
            'name': form.get('name', ''),
            'title': form.get('title', ''),
            'bio': form.get('bio', ''),
            'email': form.get('email', ''),
            'phone': form.get('phone', ''),
            'location': form.get('location', ''),
            'skills': skills_str,
            'experience': experience_str,
            'education': education_str,
            'projects': projects_str,
            'links': links,
        }
        session['portfolio_data'] = data
        return render_template('portfolio_template.html', **data)
    return render_template('portfolio_generator.html')

@app.route('/portfolio-preview')
def portfolio_preview():
    data = session.get('portfolio_data')
    if not data:
        return redirect(url_for('portfolio_generator'))
    return render_template('portfolio_template.html', **data)

@app.route('/portfolio-export-zip')
def portfolio_export_zip():
    import tempfile
    data = session.get('portfolio_data')
    if not data:
        return redirect(url_for('portfolio_generator'))
    html = render_template('portfolio_template.html', **data)
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, 'index.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        import shutil
        assets_src = os.path.join('AI Portfolio', 'assets')
        assets_dst = os.path.join(tmpdir, 'assets')
        if os.path.exists(assets_src):
            shutil.copytree(assets_src, assets_dst)
        with open(html_path, 'r+', encoding='utf-8') as f:
            html_data = f.read()
            html_data = html_data.replace('/assets/', 'assets/')
            f.seek(0)
            f.write(html_data)
            f.truncate()
        zip_path = os.path.join(tmpdir, 'portfolio_site.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(html_path, arcname='index.html')
            for root, dirs, files in os.walk(assets_dst):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmpdir)
                    zipf.write(file_path, arcname=arcname)
        with open(zip_path, 'rb') as f:
            response = make_response(f.read())
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Content-Disposition'] = 'attachment; filename=portfolio_site.zip'
            return response

if __name__ == "__main__":
    app.run(debug=True)