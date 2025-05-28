# ♻️ Trash Detection App - YOLOv8

A lightweight waste classification and detection app built using **YOLOv8m** and **Streamlit**, designed to help detect recyclable waste in real-time from images. This project is intended as a prototype for improving environmental waste management using computer vision.

---

## 📦 Model Overview

| Component      | Details                       |
| -------------- | ----------------------------- |
| Model          | YOLOv8m (Ultralytics)         |
| Epochs         | 150                           |
| Input Size     | 1024 x 1024                   |
| Dataset Source | Roboflow & Kaggle Dataset     |
| Trained On     | Google Colab (T4) + Local GPU |

### 🔍 Performance Summary

| Metric           | Value  |
| ---------------- | ------ |
| mAP\@0.5         | \~0.57 |
| mAP\@0.5:0.95    | \~0.34 |
| Best Epoch       | \~112  |
| Detected Classes | 10     |

> 📉 **Note:** The model shows strong detection on dominant classes (e.g., plastic bottles, paper, cans), but performance drops significantly on underrepresented classes like batteries or styrofoam due to class imbalance and limited examples.

---

## 🖼️ Supported Classes

* Plastic Bottle
* Plastic Bag
* Paper/Cardboard
* Drink Carton
* Metal Can
* Glass Bottle
* Pop Tab
* Bottle Cap
* Battery

## 💻 How to Run Locally

```bash
# 1. Clone the repo
https://github.com/your-username/trash-detection-app.git
cd trash-detection-app

# 2. (Optional) Create a virtual environment
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

---

## 🧠 Features

* Real-time image upload for detection
* Customizable confidence threshold
* Annotated image results
* Explanation and fallback guidance if detection fails
* Ready to deploy to Streamlit Cloud

---

## 🚧 Known Limitations

* **Model overfits to dominant classes**
* **May misclassify visually similar objects** (e.g. bottle → carton)
* Detection quality drops for low-resolution or cluttered backgrounds

---

## 🌐 Deployment (Streamlit Cloud)

This app can be deployed instantly via Streamlit Cloud:

1. Push your code to GitHub
2. Connect your GitHub repo to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Set main file as `app.py`

---

## 🤝 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [TACO Dataset](https://tacodataset.org/)
* [Roboflow Augmentation Platform](https://roboflow.com/)

> Created with ❤️ by [@wooffyy](https://github.com/wooffyy)
