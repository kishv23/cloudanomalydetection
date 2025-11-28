import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import base64

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Cloud Anomaly Detection | Enterprise",
    page_icon="🛰️",
    layout="wide"
)

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

if "vm_count" not in st.session_state:
    st.session_state.vm_count = 5

if "refresh_rate" not in st.session_state:
    st.session_state.refresh_rate = 3


# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    try:
        scaler = joblib.load("scaler.pkl")
    except:
        scaler = None
    return model, label_enc, scaler

model, label_enc, scaler = load_artifacts()


# ===============================
# DASHBOARD HEADER
# ===============================
st.markdown(
    """
    <h1 style='text-align:center;color:#2f80ed;'>
        🛰️ Enterprise Cloud Anomaly Detection System
    </h1>
    <h4 style='text-align:center;color:gray;margin-top:-10px;'>
        Real-Time Monitoring • Auto Reporting
    </h4><br>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ Global Settings")

# Controls
vm_count = st.sidebar.selectbox("Number of VMs", [3,5,7,10], index=1)
refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 1, 10, 3)

st.session_state.vm_count = vm_count
st.session_state.refresh_rate = refresh_rate


# ===============================
# MATPLOTLIB CHART GENERATORS
# ===============================
def create_anomaly_chart(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["prediction"], color="red", label="Anomaly (1=Yes,0=No)")
    ax.set_title("Anomaly Trend")
    ax.set_ylabel("Prediction")
    ax.set_xlabel("Samples")
    ax.grid(True)

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer.getvalue()


def create_cpu_chart(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["cpu_usage"], color="blue", label="CPU Usage")
    ax.set_title("CPU Usage Trend")
    ax.set_ylabel("CPU %")
    ax.set_xlabel("Samples")
    ax.grid(True)

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer.getvalue()


# ===============================
# HTML REPORT GENERATION
# ===============================
def generate_html_report(df, chart1_png, chart2_png):

    chart1_b64 = base64.b64encode(chart1_png).decode()
    chart2_b64 = base64.b64encode(chart2_png).decode()

    anomalies = df[df["prediction"] == 1]
    anomaly_rate = len(anomalies) / len(df) * 100 if len(df) else 0

    html = f"""
    <html>
    <body style="font-family: Arial; padding: 20px;">

    <h1>Cloud Anomaly Detection Report</h1>

    <h2>Summary</h2>
    <p><b>Total Samples:</b> {len(df)}</p>
    <p><b>Total Anomalies:</b> {len(anomalies)}</p>
    <p><b>Anomaly Rate:</b> {anomaly_rate:.2f}%</p>

    <h2>Charts</h2>

    <h3>Anomaly Trend</h3>
    <img width="600" src="data:image/png;base64,{chart1_b64}">

    <h3>CPU Usage Trend</h3>
    <img width="600" src="data:image/png;base64,{chart2_b64}">

    <h2>Recent Data</h2>
    {df.tail(10).to_html(index=False)}

    </body>
    </html>
    """
    return html


# ===============================
# PDF REPORT GENERATION
# ===============================
def generate_pdf_report(df, chart1_png, chart2_png):

    anomalies = df[df["prediction"] == 1]
    anomaly_rate = (len(anomalies)/len(df))*100 if len(df) else 0

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Cloud Anomaly Detection Report", styles["Title"]))
    story.append(Spacer(1, 12))

    summary = f"""
    Total Samples: {len(df)}<br/>
    Total Anomalies: {len(anomalies)}<br/>
    Anomaly Rate: {anomaly_rate:.2f}%<br/>
    """
    story.append(Paragraph(summary, styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Anomaly Trend", styles["Heading2"]))
    img1 = Image(BytesIO(chart1_png))
    img1._restrictSize(400, 200)
    story.append(img1)
    story.append(Spacer(1, 20))

    story.append(Paragraph("CPU Usage Trend", styles["Heading2"]))
    img2 = Image(BytesIO(chart2_png))
    img2._restrictSize(400, 200)
    story.append(img2)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ===============================
# DOWNLOAD SECTION
# ===============================
st.markdown("---")
st.header("📄 Auto-Generated Reports")

if len(st.session_state.history) == 0:
    st.warning("No data available yet.")
else:
    df = pd.DataFrame(st.session_state.history)

    chart1_png = create_anomaly_chart(df)
    chart2_png = create_cpu_chart(df)

    html_report = generate_html_report(df, chart1_png, chart2_png)
    pdf_report = generate_pdf_report(df, chart1_png, chart2_png)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button("⬇️ Download HTML Report", data=html_report,
                           file_name="cloud_report.html", mime="text/html")

    with col2:
        st.download_button("⬇️ Download PDF Report", data=pdf_report,
                           file_name="cloud_report.pdf", mime="application/pdf")


# ===============================
# EMAIL FUNCTIONALITY
# ===============================
st.header("📧 Email Auto-Generated Report")

email_to = st.text_input("Recipient Email Address")

def send_email(receiver, pdf_bytes, html_text):

    sender = "kish.vish377@gmail.com"
    password = "hdft kduu ojdb yacu"  

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = "Cloud Anomaly Detection Report"

    msg.attach(MIMEText("Attached is your automatic cloud anomaly report."))

    # Attach PDF
    part = MIMEApplication(pdf_bytes, _subtype="pdf")
    part.add_header("Content-Disposition", "attachment", filename="cloud_report.pdf")
    msg.attach(part)

    # Attach HTML
    part2 = MIMEApplication(html_text.encode(), _subtype="html")
    part2.add_header("Content-Disposition", "attachment", filename="cloud_report.html")
    msg.attach(part2)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        return str(e)


if st.button("Send Email Report"):
    if not email_to.strip():
        st.warning("Enter a valid email address.")
    else:
        success = send_email(email_to, pdf_report, html_report)
        if success == True:
            st.success("📨 Email sent successfully!")
        else:
            st.error(f"❌ Failed to send: {success}")
