# app.py
# Haruf-e-Tahajji — Full Letters Kid App (Updated with clickable images and detailed Urdu instructions)
# Usage:
# pip install streamlit librosa scikit-learn joblib numpy scipy pillow
# streamlit run app.py

import os
import time
import json
import glob
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
load_dotenv()  # This will load environment variables from .env file


# -----------------------
# Load Makhraj & Tajweed Rules
# -----------------------
with open("haruf_rules.json", encoding="utf-8") as f:
    HARUF_RULES = json.load(f)

# -----------------------
# Config / paths / labels
# -----------------------
DATA_DIR = "recordings"
MODEL_PATH = "haruf_model.joblib"
META_PATH = "meta.json"
PROGRESS_PATH = "progress.json"
ADMIN_PASSWORD = os.getenv("MY_APP_PASSWORD")


LETTER_KEYS = [
    "alif","baa","taa","thaa","jeem","haa","kha","dal","dhal","ra","zay",
    "seen","sheen","saad","daad","tta","zza","ain","ghain","fa","qaf","kaf",
    "lam","meem","noon","ha","waw","ya"
]

# Default Urdu feedback mapping (detailed)
DEFAULT_URDU_LABELS = {
    "alif": "الف — آواز نرمی سے گلے سے نکالیں، زبان اوپر نہ لگائیں، صاف اور آہستہ پڑھیں۔",
    "baa": "ب — ہونٹ مکمل بند کریں، ہوا کو باہر نکالیں، صاف آواز کے ساتھ پڑھیں۔",
    "taa": "ت — زبان سامنے کے دانتوں کے پیچھے لگائیں، صاف اور ہلکی آواز نکالیں۔",
    "thaa": "ث — دانتوں کے درمیان سے ہلکی ہوا نکالیں، زور نہ دیں۔",
    "jeem": "ج — زبان کے پچھلے حصے کو نرم تالو سے لگائیں، صاف آواز کے ساتھ پڑھیں۔",
    "haa": "ح — گلے سے نرم آواز نکالیں، زور نہ دیں، آہستہ پڑھیں۔",
    "kha": "خ — گلے کے پچھلے حصے سے آواز نکالیں، صاف اور گہری آواز کے ساتھ۔",
    "dal": "د — زبان سامنے کے حصے پر ہلکا ٹچ کریں، صاف پڑھیں۔",
    "dhal": "ذ — زبان د کے قریب، مگر نرم لہجہ رکھیں۔",
    "ra": "ر — زبان کے سر کو ہلکا سا ہلائیں، صاف آواز آئے گی۔",
    "zay": "ز — دانتوں کے پیچھے کی آواز صاف کریں۔",
    "seen": "س — س کی سیٹی جیسی آواز لائیں، نرم اور واضح۔",
    "sheen": "ش — نرم ہوا کے بہاؤ کے ساتھ پڑھیں۔",
    "saad": "ص — زور دار مگر صاف آواز نکالیں۔",
    "daad": "ض — دیگر حروف سے فرق رکھیں، تھوڑا گہرا کریں۔",
    "tta": "ط — زور دار، مگر صاف آواز نکالیں۔",
    "zza": "ظ — آہستہ اور واضح آواز کے ساتھ پڑھیں۔",
    "ain": "ع — گہرائی میں آواز آتی ہے، نرمی سے ادا کریں۔",
    "ghain": "غ — گلے کے پچھلے حصے سے آواز نکالیں، ذرا زور دیں۔",
    "fa": "ف — دانتوں کے اوپر ہلکا سا لمس رکھیں، ہوا باہر نکالیں۔",
    "qaf": "ق — گہرے حصے سے آواز نکالیں، احتیاط کے ساتھ۔",
    "kaf": "ک — پچھلے زبانی حصہ استعمال کریں، صاف رہے گا۔",
    "lam": "ل — زبان کے سامنے کا حصہ چھوتا ہے، واضح رکھیں۔",
    "meem": "م — ہونٹ بند کر کے نالہ دار غنہ برقرار رکھیں۔",
    "noon": "ن — ناک سے تھوڑی سی گونج آئے، غنہ صحیح کریں۔",
    "ha": "ہ — نرم سانس جیسی آواز، آہستہ کریں۔",
    "waw": "و — لبوں کو گول کر کے آواز نکالیں۔",
    "ya": "ی — زبان کی پوزیشن کم اوپر، واضح آواز نکالیں۔"
}

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(META_PATH):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"urdu": DEFAULT_URDU_LABELS}, f, ensure_ascii=False, indent=2)
if not os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
# Ensure folders for all letters exist
for letter in LETTER_KEYS:
    os.makedirs(os.path.join(DATA_DIR, letter), exist_ok=True)

# -----------------------
# Feature extraction (without recording)
# -----------------------
def extract_mfcc_features_from_array(audio_array, sr=22050, n_mfcc=13):
    y_trim = librosa.effects.trim(audio_array, top_db=30)[0]
    if y_trim.shape[0] == 0:
        y_trim = audio_array
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def load_dataset_from_recordings():
    X, y = [], []
    for label_dir in sorted(os.listdir(DATA_DIR)):
        p = os.path.join(DATA_DIR, label_dir)
        if os.path.isdir(p):
            for w in glob.glob(os.path.join(p, "*.wav")):
                try:
                    y_array, _ = librosa.load(w, sr=22050)
                    feats = extract_mfcc_features_from_array(y_array)
                    X.append(feats)
                    y.append(label_dir)
                except:
                    continue
    if not X:
        return None, None
    return np.vstack(X), np.array(y)

# -----------------------
# Meta helpers
# -----------------------
def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(meta):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def get_urdu_for_label(label):
    meta = load_meta()
    return meta.get("urdu", {}).get(label, f"آپ نے {label} پڑھا — ماڈل غیر یقینی ہے۔")

def load_progress():
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

# -----------------------
# Letter image (Hijjah) clickable
# -----------------------
def make_letter_image(letter_key, big=False):
    W,H = (240,240) if big else (140,140)
    img = Image.new("RGBA", (W,H), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", 96 if big else 56)
    except:
        fnt = ImageFont.load_default()

    arabic_hijjah_map = {
        "alif":"اَ","baa":"بَ","taa":"تَ","thaa":"ثَ","jeem":"جَ","haa":"حَ",
        "kha":"خَ","dal":"دَ","dhal":"ذَ","ra":"رَ","zay":"زَ","seen":"سَ",
        "sheen":"شَ","saad":"صَ","daad":"ضَ","tta":"طَ","zza":"ظَ","ain":"عَ",
        "ghain":"غَ","fa":"فَ","qaf":"قَ","kaf":"کَ","lam":"لَ","meem":"مَ",
        "noon":"نَ","ha":"ہَ","waw":"وَ","ya":"یَ"
    }
    glyph = arabic_hijjah_map.get(letter_key, letter_key)
    bbox = draw.textbbox((0,0), glyph, font=fnt)
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    draw.text(((W-w)/2,(H-h)/2), glyph, font=fnt, fill=(20,20,20))
    return img

# -----------------------
# Streamlit Layout
# -----------------------
st.set_page_config(page_title="Haruf-e-Tahajji for Kids", layout="wide")
st.title("Haruf-e-Tahajji — Kids Trainer (Full Letters)")
st.markdown("This app predict alphabet by frequency so sometimes it can be mistake— Made with ❤️ — Haruf-e-Tahajji Kids Trainer.")

if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

mode = st.sidebar.radio("Mode", ["Home","Practice","Admin","Train","Manage/Export"])

if mode in ["Admin","Train","Manage/Export"] and not st.session_state.admin_logged_in:
    pwd = st.sidebar.text_input("Enter Admin Password", type="password")
    if st.sidebar.button("Login"):
        if pwd==ADMIN_PASSWORD:
            st.session_state.admin_logged_in=True
            st.sidebar.success("Admin Access Granted")
        else:
            st.sidebar.error("Wrong Password")
    st.stop()

if st.session_state.admin_logged_in and st.sidebar.button("Logout Admin"):
    st.session_state.admin_logged_in=False
    st.sidebar.success("Logged out")

# -----------------------
# Home Page — Images + Audio + Adaiygi on IMAGE click
# -----------------------
if mode == "Home":
    st.header("حروفِ تہجی")
    st.markdown("تصویر دیکھیں، ▶️ دبائیں اور صحیح آواز سنیں")

    AUDIO_DIR = os.path.join(DATA_DIR, "letter_audio")
    progress = load_progress()

    PER_ROW = 4  # mobile friendly

    for i in range(0, len(LETTER_KEYS), PER_ROW):
        row = LETTER_KEYS[i:i+PER_ROW]
        cols = st.columns(len(row))

        for col, key in zip(cols, row):
            with col:
                # IMAGE
                img_path = os.path.join(DATA_DIR, "letter_images", f"{key}.png")
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.image(make_letter_image(key), use_container_width=True)

                # PLAY AUDIO
                audio_path = os.path.join(AUDIO_DIR, f"{key}.wav")
                if os.path.exists(audio_path):
                    st.audio(audio_path)
                else:
                    st.caption("🔇 آواز موجود نہیں")

                # PRONUNCIATION INFO
                if st.button("ادائیگی کا طریقہ", key=f"info_{key}"):
                    st.info(get_urdu_for_label(key))

                # STARS
                st.markdown("⭐" * progress.get(key, 0))


# -----------------------
# Admin
# -----------------------
elif mode=="Admin":
    st.header("Admin — Letters & Record Samples")
    st.markdown("Admin: حروف پہلے سے تیار ہیں۔ مزید labels بنا سکتے ہیں یا recording کر سکتے ہیں۔")
    for k in LETTER_KEYS:
        os.makedirs(os.path.join(DATA_DIR,k),exist_ok=True)

    labels=[d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR,d))]
    chosen=st.selectbox("Choose letter",labels,index=labels.index("alif") if "alif" in labels else 0)
    duration=st.slider("Duration (seconds)",0.6,2.5,1.2,0.1)
    col1,col2=st.columns([2,1])
    with col1:
        if st.button("Record sample"):
            fs,a=record_audio(duration=duration,fs=22050)
            fname=f"{chosen}_{int(time.time())}.wav"
            p=os.path.join(DATA_DIR,chosen,fname)
            save_wav_from_array(p,fs,a)
            st.success(f"Saved: {p}")
            st.audio(p)
    with col2:
        cnt=len(glob.glob(os.path.join(DATA_DIR,chosen,"*.wav")))
        st.write(f"{chosen}: {cnt} samples")

    new_label=st.text_input("New label key","")
    new_label_urdu=st.text_input("Urdu feedback (optional)","")
    if st.button("Create label folder"):
        if new_label.strip():
            os.makedirs(os.path.join(DATA_DIR,new_label.strip()),exist_ok=True)
            meta=load_meta()
            if new_label_urdu.strip():
                meta.setdefault("urdu",{})[new_label.strip()]=new_label_urdu.strip()
                save_meta(meta)
            st.success("Label created")
        else:
            st.error("Label name required")
# -----------------------
# Admin: Upload Letter Images
# -----------------------
if mode == "Admin":
    st.header("Admin — Letters & Record Samples / Images")
    
    # Create folders if not exist
    for k in LETTER_KEYS:
        os.makedirs(os.path.join(DATA_DIR, k), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "letter_images"), exist_ok=True)

    st.subheader("Upload letter image")
    selected_label = st.selectbox("Select letter to upload image", LETTER_KEYS)
    uploaded_file = st.file_uploader("Choose PNG/JPG image", type=["png","jpg","jpeg"], key="img_upload")

    if uploaded_file is not None:
        save_path = os.path.join(DATA_DIR, "letter_images", f"{selected_label}.png")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Image for '{selected_label}' uploaded successfully")
# -----------------------
# Admin: Upload Letter Audio
# -----------------------
if mode == "Admin":
    st.header("Admin — Upload Letter Audio")

    AUDIO_DIR = os.path.join(DATA_DIR, "letter_audio")
    os.makedirs(AUDIO_DIR, exist_ok=True)

    selected_letter = st.selectbox("حرف منتخب کریں", LETTER_KEYS)

    audio_file = st.file_uploader(
        "اس حرف کی صحیح ادائیگی کی آواز upload کریں (WAV / MP3)",
        type=["wav", "mp3"]
    )

    if audio_file is not None:
        audio_path = os.path.join(AUDIO_DIR, f"{selected_letter}.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        st.success(f"✅ {selected_letter} کی آواز محفوظ ہو گئی")

# -----------------------
# Train
# -----------------------
elif mode=="Train":
    st.header("Train model")
    if st.button("Preview dataset"):
        X,y=load_dataset_from_recordings()
        if X is None:
            st.error("کوئی ریکارڈنگ نہیں ملی")
        else:
            st.success(f"{X.shape[0]} samples, {len(np.unique(y))} labels")
            st.write("Labels:",list(np.unique(y)))
            st.write("Feature vector shape:",X.shape[1])
    test_frac=st.slider("Test set %",5,40,20)
    n_estimators=st.slider("RandomForest trees",10,200,80)
    if st.button("Train now"):
        X,y=load_dataset_from_recordings()
        if X is None: st.error("کوئی ڈیٹا نہیں")
        else:
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_frac/100.0,random_state=42,stratify=y)
            st.info("Training ...")
            clf=RandomForestClassifier(n_estimators=n_estimators,random_state=42)
            clf.fit(X_train,y_train)
            preds=clf.predict(X_test)
            acc=accuracy_score(y_test,preds)
            st.success(f"Training done — test accuracy: {acc:.3f}")
            st.text(classification_report(y_test,preds))
            joblib.dump(clf,MODEL_PATH)
            meta=load_meta()
            meta["labels"]=list(np.unique(y))
            save_meta(meta)

# -----------------------
# Practice
# -----------------------
elif mode=="Practice":
    st.header("Practice — Record & Check")
    if not os.path.exists(MODEL_PATH):
        st.warning("ماڈل موجود نہیں — پہلے Train کریں")
    else:
        try: clf=joblib.load(MODEL_PATH)
        except Exception as e: st.error(f"Load error: {e}"); st.stop()
        BASE_DURATION=2.0
        st.markdown("### 🎤 بڑا بٹن دبائیں اور حرف پڑھیں")

        if st.button("🎤 Record & Check"):
            try:
                fs,a=record_audio(duration=BASE_DURATION,fs=22050)
                new_duration=adaptive_duration(a,base=BASE_DURATION)
                if new_duration>BASE_DURATION:
                    st.info("🔁 آواز تھوڑی چھوٹی تھی، دوبارہ سن رہے ہیں")
                    fs,a=record_audio(duration=new_duration,fs=22050)
                tmp=f"tmp_{int(time.time())}.wav"
                save_wav_from_array(tmp,fs,a)
                st.audio(tmp)
                feats=extract_mfcc_features_from_array(a,sr=22050)
                probs=clf.predict_proba([feats])[0]
                labels=clf.classes_
                top=np.argmax(probs)
                top_label=labels[top]
                conf=float(probs[top])
                urdu=get_urdu_for_label(top_label)

                if conf>=0.8: teacher_msg=urdu
                elif conf>=0.6: teacher_msg="کوشش اچھی ہے — تھوڑی سی درستگی درکار ہے۔"
                elif conf>=0.4: teacher_msg="غلط مخارج — دوبارہ آہستہ پڑھیں۔"
                else: teacher_msg="آواز واضح نہیں تھی — دوبارہ کوشش کریں۔"

                rule_data=HARUF_RULES.get(top_label,{})
                letter_type=rule_data.get("type","light")
                if letter_type=="heavy" and conf<0.75:
                    teacher_msg="❌ یہ حرف بھاری ہے، زور کے ساتھ پڑھیں۔"
                elif letter_type=="light" and conf<0.75:
                    teacher_msg="❌ یہ حرف ہلکا ہے، زور نہ دیں۔"

                st.success(f"حرف: {top_label} | اعتماد: {conf:.2f}")
                st.markdown(f"### 🧑‍🏫 استاد کا پیغام:\n**{teacher_msg}**")

                if conf>=0.8:
                    progress=load_progress()
                    progress[top_label]=min(3,progress.get(top_label,0)+1)
                    save_progress(progress)
                    st.balloons()
                    st.markdown("## ⭐ آپ کو ستارہ ملا ⭐")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# -----------------------
# Manage / Export
# -----------------------
elif mode=="Manage/Export":
    st.header("Manage / Export — Samples, Images & Audio")

    # ======================
    # 1️⃣ Recorded Samples
    # ======================
    st.subheader("🎤 Recorded Samples")

    labels = [d for d in sorted(os.listdir(DATA_DIR))
              if os.path.isdir(os.path.join(DATA_DIR, d))]

    if not labels:
        st.info("No labels available")
    else:
        selected_label = st.selectbox("Select letter", labels)
        files = sorted(glob.glob(os.path.join(DATA_DIR, selected_label, "*.wav")))

        if not files:
            st.warning("No recordings for this letter")
        else:
            selected_file = st.selectbox(
                "Select recording",
                files,
                format_func=lambda x: os.path.basename(x)
            )
            st.audio(selected_file)

            if st.button("🗑️ Delete Sample"):
                os.remove(selected_file)
                st.success("✅ Sample deleted")
                st.experimental_rerun()

    st.markdown("---")

    # ======================
    # 2️⃣ Letter Images
    # ======================
    st.subheader("🖼️ Letter Images")

    IMAGE_DIR = os.path.join(DATA_DIR, "letter_images")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))

    if images:
        img_choice = st.selectbox(
            "Select Picture",
            images,
            format_func=lambda x: os.path.basename(x),
            key="img_del"
        )
        st.image(img_choice, width=150)

        if st.button("🗑️ Delete Picture"):
            os.remove(img_choice)
            st.success("✅ Picture Deleted")
            st.experimental_rerun()
    else:
        st.info("No any picture found")

    st.markdown("---")

    # ======================
    # 3️⃣ Letter Audio
    # ======================
    st.subheader("🎧 Letter Audio")

    AUDIO_DIR = os.path.join(DATA_DIR, "letter_audio")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    audios = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

    if audios:
        audio_choice = st.selectbox(
            "Select Audio",
            audios,
            format_func=lambda x: os.path.basename(x),
            key="audio_del"
        )
        st.audio(audio_choice)

        if st.button("🗑️ Delete Audio"):
            os.remove(audio_choice)
            st.success("✅ Audio Deleted")
            st.experimental_rerun()
    else:
        st.info("No any audio file found")
