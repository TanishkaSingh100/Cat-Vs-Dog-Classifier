# Cat-Vs-Dog-Classifier

This app allows users to upload an image of a cat or dog and instantly get a prediction from the model, along with a confidence score and an engaging visual display.

The classifier was trained on the [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) cats_vs_dogs dataset using a custom CNN architecture.


## Screenshot : 

Home Page :
[Home Page](https://github.com/TanishkaSingh100/Cat-Vs-Dog-Classifier/blob/main/screenshots/home.png)

Fun Mode Output :
[Fun Mode](https://github.com/TanishkaSingh100/Cat-Vs-Dog-Classifier/blob/main/screenshots/fun_mode_output.png)

Basic Mode Output :
[Basic Mode](https://github.com/TanishkaSingh100/Cat-Vs-Dog-Classifier/blob/main/screenshots/basic_mode_output.png)

---

# Model Details

| Metric               | Value       |
|----------------------|-------------|
| Training Accuracy    | 94.22%      |
| Training Loss        | 0.1332      |
| Validation Accuracy  | 80.48%      |

- *Architecture:* Convolutional Neural Network (CNN)
- *Input Size:* 256×256×3 (RGB images)
- *Output:* Binary classification — Cat or Dog 
- *Final Activation:* Sigmoid
- *Optimizer:* Adam

---

# Features

- Upload JPG, JPEG, or PNG images for classification
- Two modes:  
  - *Fun Mode* with colorful confidence bars and emojis  
  - *Basic Mode* with simple text results
- Sidebar with model information and credits
- Responsive design for smooth user experience

---

# How to Run Locally

1. Clone the repo:

   git clone https://github.com/TanishkaSingh100/Cat-Vs-Dog-Classifier.git

2. Install dependencies:

   pip install -r requirements.txt

3. Run the Streamlit app:

   streamlit run app.py

---

# Deployment Status

Currently working on deployment via Hugging Face Spaces. Will update the live demo link soon!

---

# Author

Tanishka Singh

---
