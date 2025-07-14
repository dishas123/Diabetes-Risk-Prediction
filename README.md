# 🩺 Diabetes Risk Assessment Tool

This is a Streamlit-based web application that predicts the risk of diabetes using health parameters such as glucose level, BMI, insulin, age, and more. It is powered by a logistic regression model trained on the Pima Indians Diabetes Dataset.

> ⚠️ **Note**: This tool is for informational purposes only and does not substitute professional medical advice.

---

## 🚀 Features

- 🔍 **Diabetes Risk Prediction** based on 8 medical attributes
- 📈 **Visual Probability Meter** (Progress bar)
- 🟢 **Risk Level Indicator** (Low / Moderate / High)
- 🩺 **Personalized Health Recommendations**
- 📊 **Feature Importance Visualization** using odds ratio
- 📚 **Educational Insight** into each contributing factor
- 🤖 **Model Evaluation & Explainability** (The Confusion matrix and the ROC Curve is displayed in the website along with the % accuracy of the logistic regression model) <br>

---

## 💡 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Model**: Logistic Regression (`sklearn.linear_model`)
- **Data Preprocessing**:
  - `pandas`, `numpy`, `SimpleImputer`, `StandardScaler`
- **Data**: Pima Indians Diabetes Dataset (`diabetes.csv`)

---

## 🧠 How It Works

1. **Data Preprocessing**:
   - Missing or zero values in columns like Glucose, BMI, etc., are imputed using the mean.
   - Features are standardized for model training.

2. **Model Training**:
   - Logistic Regression model is trained to classify diabetic vs. non-diabetic.
   - Caching is used to prevent retraining on every run.

3. **User Input via Sidebar**:
   - Input fields for:
     - Pregnancies
     - Glucose
     - Blood Pressure
     - Skin Thickness
     - Insulin
     - BMI
     - Diabetes Pedigree Function
     - Age

4. **Risk Assessment**:
   - The model outputs a **probability of diabetes**.
   - Results include a visual meter, risk category, and tailored health tips.

5. **Explanation & Education**:
   - Bar chart shows the **impact of each feature**.
   - Written explanations help users understand the role of each factor.

---

## 📷 Screenshot

**IMAGE 1 :** <br>
<br>
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/29bf4a96-2cfe-4302-81a4-7dbb9b2c929f" />
<br>
<br>
**IMAGE 2 :** <br>
<br>
<img width="839" height="1014" alt="image" src="https://github.com/user-attachments/assets/57df34ce-1830-419b-9e6f-a94585583527" />
<br>



---

## 🏁 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/dishas123/Diabetes-Risk-Prediction.git
cd Diabetes-Risk-Prediction

Run the app:
streamlit run app.py
