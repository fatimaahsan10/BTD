import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import qrcode
from gtts import gTTS
import os

# ✅ Load Model
MODEL_PATH = "brain_tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Multi-language Translations
translations = {
    "en": {
        "title": "🧠 Brain Tumor Detection App",
        "desc": "**Early Detection Saves Lives!** Upload MRI, enter details, and download your report.",
        "upload": "Upload MRI Image",
        "name": "Patient Name",
        "age": "Patient Age",
        "email": "Email (optional)",
        "analyze": "🔍 Analyze Image",
        "result": "Prediction Result",
        "report": "Download Report (PDF)",
        "premium": "Premium Feature",
        "tts_message": "Your result is:",
        "premium_text": "For Premium Users: AI Health Assistant can schedule an appointment with a doctor!",
        "no_premium": "No premium action required.",
        "advice_yes": "Consult a neurologist immediately.",
        "advice_no": "No tumor detected. Stay healthy!",
        "report_title": "Brain Tumor Detection Report",
        "email_msg": "Report sent to {email} (simulation)"
    },
    "ur": {
        "title": "🧠 دماغی ٹیومر پتہ لگانے والا ایپ",
        "desc": "**جلد پتہ لگانا زندگی بچاتا ہے!** MRI اپلوڈ کریں اور رپورٹ حاصل کریں۔",
        "upload": "MRI تصویر اپلوڈ کریں",
        "name": "مریض کا نام",
        "age": "مریض کی عمر",
        "email": "ای میل (اختیاری)",
        "analyze": "🔍 تصویر کا تجزیہ کریں",
        "result": "نتیجہ",
        "report": "رپورٹ ڈاؤن لوڈ کریں",
        "premium": "پریمیئم فیچر",
        "tts_message": "آپ کا نتیجہ یہ ہے:",
        "premium_text": "پریمیئم یوزرز کے لیے: ڈاکٹر سے اپائنٹمنٹ شیڈول کی جا سکتی ہے!",
        "no_premium": "کوئی پریمیئم ایکشن نہیں۔",
        "advice_yes": "فوراً نیورولوجسٹ سے رجوع کریں۔",
        "advice_no": "کوئی ٹیومر نہیں ملا۔ صحت مند رہیں!",
        "report_title": "دماغی ٹیومر رپورٹ",
        "email_msg": "{email} پر رپورٹ بھیجی گئی (سیمولیشن)"
    }
}

# ✅ PDF Report with QR Code
def generate_pdf(name, age, result, confidence, lang):
    file_path = tempfile.mktemp(suffix=".pdf")
    qr_img_path = tempfile.mktemp(suffix=".png")

    # Generate QR Code for Hugging Face app link
    qr = qrcode.QRCode()
    qr.add_data("https://huggingface.co/spaces/your-space-name")  # Replace with your Hugging Face app link
    qr.make()
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save(qr_img_path)

    # Create PDF
    c = canvas.Canvas(file_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, 750, translations[lang]["report_title"])
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"{translations[lang]['name']}: {name}")
    c.drawString(50, 680, f"{translations[lang]['age']}: {age}")
    c.drawString(50, 660, f"{translations[lang]['result']}: {result}")
    c.drawString(50, 640, f"Confidence: {confidence:.2f}%")
    c.drawString(50, 620, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    advice = translations[lang]["advice_yes"] if result == "Tumor" else translations[lang]["advice_no"]
    c.drawString(50, 600, f"Advice: {advice}")
    c.drawImage(qr_img_path, 400, 550, width=100, height=100)
    c.save()
    return file_path

# ✅ Prediction Function
def predict_and_report(img, name, age, email, lang):
    if img is None or not name or not age:
        return "Please provide all details.", None, None, None, None, None

    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

    # PDF with QR
    pdf_report = generate_pdf(name, age, result, confidence, lang)

    # Premium message
    premium_msg = translations[lang]["premium_text"] if result == "Tumor" else translations[lang]["no_premium"]

    # Voice output (safe)
    try:
        tts_text = f"{translations[lang]['tts_message']} {result}"
        tts = gTTS(tts_text, lang='ur' if lang == 'ur' else 'en')
        tts_path = tempfile.mktemp(suffix=".mp3")
        tts.save(tts_path)
    except:
        tts_path = None

    # Confidence Bar HTML
    color = "green" if result == "No Tumor" else "red"
    confidence_html = f"""
    <div style='width:100%;background:#ddd;border-radius:10px;'>
        <div style='width:{confidence:.2f}%;background:{color};padding:5px;border-radius:10px;color:white;text-align:center;'>
            {confidence:.2f}%
        </div>
    </div>
    """

    email_msg = translations[lang]["email_msg"].format(email=email) if email else ""

    return f"{translations[lang]['result']}: {result}", confidence_html, pdf_report, premium_msg, tts_path, email_msg

# ✅ Gradio UI
with gr.Blocks(theme="default") as demo:
    gr.Markdown("# 🧠 Brain Tumor Detection App")
    gr.Markdown("**Early Detection Saves Lives! Built for rural and underdeveloped areas.**")

    lang_selector = gr.Dropdown(choices=["en", "ur"], value="en", label="Language")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload MRI Image")
            name_input = gr.Textbox(label="Patient Name")
            age_input = gr.Textbox(label="Patient Age")
            email_input = gr.Textbox(label="Email (Optional)")
            analyze_btn = gr.Button("🔍 Analyze Image")
            gr.Markdown("[🔗 Learn More About Brain Tumors](https://www.who.int/health-topics/cancer)")

        with gr.Column():
            result_output = gr.Textbox(label="Prediction Result")
            confidence_bar = gr.HTML(label="Confidence")
            report_output = gr.File(label="Download Report (PDF)")
            premium_output = gr.Textbox(label="Premium Feature")
            audio_output = gr.Audio(label="Voice Output")
            email_output = gr.Textbox(label="Email Status")

    # Educational Tips Section
    gr.Markdown("### 📢 Why Early Detection Matters?")
    gr.Markdown("**90% of brain tumors can be treated if detected early. This app helps rural areas get faster diagnoses.**")

    # Appointment Simulation
    appointment_btn = gr.Button("📅 Book Appointment (Premium)")
    appointment_btn.click(lambda: "This feature is available for Premium Users.", outputs=premium_output)

    # Actions
    analyze_btn.click(
        predict_and_report,
        inputs=[img_input, name_input, age_input, email_input, lang_selector],
        outputs=[result_output, confidence_bar, report_output, premium_output, audio_output, email_output]
    )

demo.launch()
