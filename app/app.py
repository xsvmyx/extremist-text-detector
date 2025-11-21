import streamlit as st
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import joblib

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("xsvmyx/extremist-text-detector")

    clf_path = hf_hub_download(repo_id="xsvmyx/extremist-text-detector", filename="clf.pkl")
    le_path = hf_hub_download(repo_id="xsvmyx/extremist-text-detector", filename="label_encoder.pkl")

    clf = joblib.load(clf_path)
    le = joblib.load(le_path)

    return embedder, clf, le

embedder, clf, le = load_models()

st.title("🛡️ Extremism Detector")
st.write("Enter a sentence and the model will predict whether it is extremist or not.")

text = st.text_area("Your sentence:", "")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        # Embedding
        emb = embedder.encode([text])

        # Prediction
        pred = clf.predict(emb)[0]
        label = le.inverse_transform([pred])[0]

        st.subheader("Result:")
        if label.lower() == "extremist":
            st.error("This sentence is classified as **extremist**.")
        else:
            st.success("This sentence is **not extremist**.")
