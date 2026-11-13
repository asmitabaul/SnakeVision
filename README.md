# SnakeVision

AI-powered identification of 10 Indian snake species using EfficientNetB3.

## Model Details
- Architecture: EfficientNetB3
- Input resolution: 300 x 300
- Number of classes: 10
- Validation accuracy: 59.56%
- Framework: TensorFlow and Keras

## Features
- Upload snake photographs
- Top-three species predictions
- Prediction confidence scores
- Streamlit-based interface

## Installation
Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run snake_species_identifier.py

## Dataset
The project uses a curated dataset of 10 Indian snake species.
The original dataset was collected using iNaturalist.

## Disclaimer
SnakeVision is an educational deep learning prototype.
Predictions are not verified species identifications and must
not be used for handling decisions or medical emergencies.