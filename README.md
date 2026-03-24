# 👁️ Multiple Eye Disease Detection using Machine Learning

A web-based application that detects multiple eye diseases from uploaded images using a trained deep learning model.

---

## 🚀 Features

* Upload eye images through a web interface
* Detect multiple eye diseases:

  * Cataract
  * Diabetic Retinopathy
  * Glaucoma
  * Normal / Other
* Displays prediction results with descriptions
* Simple and user-friendly UI built with Flask

---

## 🧠 Tech Stack

* **Python**
* **Flask** (Web Framework)
* **TensorFlow / Keras** (Deep Learning)
* **OpenCV & PIL** (Image Processing)
* **HTML, CSS (Bulma)**

---

## 📁 Project Structure

```
eye_detection/
│
├── app.py                # Main Flask application
├── build_model.py        # Model training script
├── augment.py            # Data augmentation
├── resize.py             # Image preprocessing
├── split_test.py         # Dataset splitting
├── model.h5              # Trained model (optional)
│
├── templates/            # HTML pages
├── static/               # CSS, images
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/22f2000834/Multiple-eye-disease-detection-using-machine-learning.git
cd Multiple-eye-disease-detection-using-machine-learning
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add the model file

Place your trained model file:

```
model.h5
```

in the root directory.

> ⚠️ Note: The model file is not included due to size limits.

---

### 5. Run the application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 📸 Screenshots

*Add screenshots of your app here (highly recommended)*

---

## ⚠️ Important Notes

* Ensure `model.h5` is available before running
* Update the model path in `app.py` if needed
* This project is for educational purposes only

---

## 🔮 Future Improvements

* Improve model accuracy
* Add more eye disease classes
* Deploy on cloud (Render / AWS / Heroku)
* Add user authentication
* Real-time camera detection

---

## 👤 Author

**Mohammad Asgar Khan Dalwai**

---

## 📄 License

This project is open-source and available under the MIT License.
