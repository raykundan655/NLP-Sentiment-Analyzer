import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

from function import preprocess, makevec

st.set_page_config(
    page_title="MoodMind AI",
    layout="wide"
)

model = load_model("ReviewAnalyiser.h5")

st.title("MoodMind AI")
st.subheader("Understand the emotion hidden inside every review")

left, right = st.columns([2,1])

with left:

    review = st.text_area(
        "Write your Review",
        height=250,
        placeholder="Write your thoughts here..."
    )

    if st.button("Analyze Review"):

        clean_text = preprocess(review)

        vector = makevec(clean_text)

        vector = np.array(vector).reshape(1,300)

        val = model.predict(vector)

        score = val[0][0]

        if score >= 0.5:

            st.success("Positive Review Detected")

            st.progress(float(score))

            st.metric(
                label="Confidence Score",
                value=f"{round(score*100,2)}%"
            )

        else:

            st.warning("😠Negative Review Detected")

            st.progress(float(1-score))

            st.metric(
                label="Confidence Score",
                value=f"{round((1-score)*100,2)}%"
            )

with right:

    st.header("About")

    st.write("""
This AI model analyzes customer reviews
and predicts whether the sentiment is
positive or negative using NLP and
Deep Learning.
""")

    