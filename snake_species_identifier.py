
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SnakeVision",
    page_icon="S",
    layout="centered"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "snake_species_model.keras"
IMG_SIZE = (300, 300)

CLASS_NAMES = [
    "Banded_Kukri_Snake",
    "Checkered_Keelback",
    "Common_Krait",
    "Indian_Cobra",
    "Indian_Python",
    "Indian_Wolf_Snake",
    "Oriental_Rat_Snake",
    "Russells_Viper",
    "Saw_scaled_Viper",
    "Trinket_Snake"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load the SnakeVision model: {e}")
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background-image:
        linear-gradient(
            rgba(244, 241, 232, 0.70),
            rgba(244, 241, 232, 0.70)
        ),
        url("https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=2400&q=85");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    max-width: 900px;
    padding: 3.5rem 3.5rem 2.5rem 3.5rem;
    background: rgba(250, 249, 243, 0.84);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 28px;
    box-shadow: 0 25px 70px rgba(25, 53, 36, 0.18);
}

h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 4rem !important;
    font-weight: 700 !important;
    color: #193524 !important;
    text-align: center;
    letter-spacing: -2px;
}

h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #244b35 !important;
}

p {
    font-family: 'DM Sans', Arial, sans-serif;
    color: #617064;
}

.subtitle {
    text-align: center;
    color: #667568;
    font-family: 'DM Sans', Arial, sans-serif;
    font-size: 1.08rem;
    font-weight: 500;
    margin-bottom: 2rem;
}

hr {
    border-color: rgba(91, 113, 96, 0.22);
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.88);
    border: 2px dashed #9db398;
    border-radius: 20px;
    padding: 28px;
}

[data-testid="stFileUploader"] button {
    background: #274f36 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-testid="stImage"] {
    border-radius: 18px;
    overflow: hidden;
}

.stProgress > div > div > div > div {
    background-color: #789461;
}

.footer {
    text-align: center;
    color: #7d897f;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    margin-top: 35px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<h1>SnakeVision</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'AI-powered identification of Indian snake species'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("### Upload an image")

st.write(
    "Choose a clear photograph of a snake to get the model's "
    "top three predictions."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # Open uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        st.divider()

        # Resize to the model's training resolution
        image_resized = image.resize(
            IMG_SIZE,
            Image.Resampling.BILINEAR
        )

        # Convert image to NumPy array
        image_array = np.array(
            image_resized,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Generate prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        # Identify top three predictions
        top3_indices = np.argsort(
            prediction[0]
        )[-3:][::-1]

        # Display results
        st.markdown("### Prediction Results")

        for i, index in enumerate(top3_indices, 1):

            species = CLASS_NAMES[index].replace(
                "_",
                " "
            )

            confidence = float(
                prediction[0][index]
            )

            with st.container(border=True):

                st.markdown(f"**Prediction #{i}**")

                st.markdown(f"### {species}")

                st.caption(
                    f"Model confidence: {confidence * 100:.2f}%"
                )

                st.progress(
                    min(max(confidence, 0.0), 1.0)
                )

        # Uncertainty notice
        top_confidence = float(
            prediction[0][top3_indices[0]]
        )

        if top_confidence < 0.50:

            st.info(
                "The model has relatively low confidence in "
                "its top prediction. Consider uploading a clearer "
                "photograph."
            )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded image: {e}"
        )


# ============================================================
# SAFETY NOTICE
# ============================================================

st.warning(
    "Important: SnakeVision is an educational deep learning "
    "project. Model predictions are not verified species "
    "identifications. Do not use them for handling decisions, "
    "determining whether a snake is venomous, or medical emergencies."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SnakeVision<br>
        Deep Learning Project • Indian Snake Species Classification
    </div>
    """,
    unsafe_allow_html=True
)
