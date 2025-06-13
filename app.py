import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Set page config FIRST
st.set_page_config(page_title="📧 AI Email Spam Detector", layout="centered")

# 🔁 Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("spam_detector_lstm.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
MAX_LEN = 300
THRESHOLD = 0.5

# 🎨 Custom Styles
st.markdown("""
    <style>
    .title {
        font-size: 2.2rem;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtext {
        font-size: 1.1rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .result-box-green {
        padding: 1rem;
        border-radius: 12px;
        background-color: #ccffcc;
        color: #006600;
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
        border: 2px solid #33cc33;
        margin-top: 1rem;
    }
    .result-box-red {
        padding: 1rem;
        border-radius: 12px;
        background-color: #ffcccc;
        color: #990000;
        font-size: 1.5rem;
        text-align: center;
        font-weight: bold;
        border: 2px solid #ff4d4d;
        margin-top: 1rem;
    }
    .box-yellow {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff8cc;
        color: #665c00;
        border: 2px dashed #ffd11a;
        margin-top: 1rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Header
st.markdown('<div class="title">📧 AI Email Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">LSTM-powered | Detects spam, phishing links, and suspicious content</div>', unsafe_allow_html=True)

# 📝 Text input and buttons
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

st.session_state.email_text = st.text_area("📨 Paste or type the email content:", value=st.session_state.email_text, height=200)
debug_mode = st.toggle("🐞 Debug Mode")

col1, col2 = st.columns([1, 1])
analyze = col1.button("🚀 Analyze Email")
reset = col2.button("🔁 Reset")

# 🔁 Reset logic
if reset:
    st.session_state.email_text = ""
    st.rerun()


# 🕵️ Suspicious Link Detection
def detect_suspicious_links(text):
    suspicious_patterns = [
        r"http[s]?://bit\.ly/\w+",
        r"http[s]?://tinyurl\.com/\w+",
        r"http[s]?://[\d\.]+",  # IP-based URL
        r"http[s]?://.*@.*",  # Email in URL
    ]
    matches = []
    for pattern in suspicious_patterns:
        matches += re.findall(pattern, text)
    return matches

# 🧨 Phishing Keyword Detection
def detect_phishing_keywords(text):
    keywords = [
        "verify your account",
        "click here",
        "login now",
        "urgent action required",
        "suspended",
        "update your information",
        "account locked",
        "security alert",
        "confirm your password"
    ]
    found = [kw for kw in keywords if kw.lower() in text.lower()]
    return found

# 🔍 Main analysis
if analyze:
    if not st.session_state.email_text.strip():
        st.warning("⚠️ Please enter email content.")
    else:
        sequence = tokenizer.texts_to_sequences([st.session_state.email_text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        pred_score = model.predict(padded)[0][0]
        prediction = "🚫 Spam" if pred_score > THRESHOLD else "✅ Not Spam"

        # 🟢 or 🔴 Display result
        if prediction == "✅ Not Spam":
            st.markdown(f'<div class="result-box-green">{prediction}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box-red">{prediction}</div>', unsafe_allow_html=True)

        # ⚠️ Suspicious content detection
        links = detect_suspicious_links(st.session_state.email_text)
        keywords = detect_phishing_keywords(st.session_state.email_text)

        if links or keywords:
            st.markdown('<div class="box-yellow">⚠️ <b>Suspicious Content Detected</b><br><ul>', unsafe_allow_html=True)
            if links:
                for link in links:
                    st.markdown(f"- Suspicious link: `{link}`", unsafe_allow_html=True)
            if keywords:
                for kw in keywords:
                    st.markdown(f"- Phishing keyword: `{kw}`", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)

        # 🐛 Debug Info
        if debug_mode:
            st.markdown("---")
            st.subheader("🔍 Debug Info")
            st.code(f"Prediction Score: {pred_score:.4f}")
            st.code(f"Threshold: {THRESHOLD}")
            st.code(f"Decision: {'Spam' if pred_score > THRESHOLD else 'Not Spam'}")
