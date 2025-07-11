import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("diabetes.csv")
    
    # Impute 0s in certain medical columns
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    
    return df

@st.cache_data
def train_model(X, y):
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predict on test data and calculate accuracy
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Additional metrics for evaluation
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    return model, scaler, test_accuracy, conf_matrix, fpr, tpr, roc_auc

def main():
    st.title("Diabetes Risk Assessment Tool")
    st.markdown("""
    This tool predicts diabetes risk based on health parameters and explains how each factor contributes to the prediction.
    **Note:** This is not medical advice. Always consult a healthcare professional.
    """)

    # Load and train
    df = load_and_preprocess_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model, scaler, test_accuracy, conf_matrix, fpr, tpr, roc_auc = train_model(X, y)

    # Sidebar input
    with st.sidebar:
        st.header("Health Parameters")
        pregnancies = st.slider("Pregnancies", 0, 17, 0)
        glucose = st.number_input("Glucose (mg/dL)", 50, 200, 100)
        bp = st.number_input("Blood Pressure (mmHg)", 20, 130, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.number_input("Insulin (μU/ml)", 0, 846, 79)
        bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.08, 2.5, 0.37)
        age = st.slider("Age", 20, 100, 30)

        st.subheader("🔍 Model Info")
        st.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")

    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                              columns=X.columns)

    if st.button("Assess Diabetes Risk"):
        try:
            scaled_input = scaler.transform(input_data)
            probability = model.predict_proba(scaled_input)[0][1] * 100
            risk_level = "High Risk" if probability >= 50 else "Moderate Risk" if probability >= 30 else "Low Risk"

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk Assessment")
                st.metric("Diabetes Probability", f"{probability:.1f}%")
                st.progress(probability / 100)

                color = "red" if risk_level == "High Risk" else "orange" if risk_level == "Moderate Risk" else "green"
                st.markdown(f"<h3 style='color:{color}'>{risk_level}</h3>", unsafe_allow_html=True)

                st.subheader("Recommendations")
                if risk_level == "High Risk":
                    st.error("""
                    - Consult a healthcare professional immediately
                    - Regular blood sugar monitoring
                    - Adopt low glycemic index diet
                    - 150 mins/week moderate exercise
                    """)
                elif risk_level == "Moderate Risk":
                    st.warning("""
                    - Regular health checkups
                    - Maintain healthy weight
                    - Reduce processed sugar intake
                    - Stress management techniques
                    """)
                else:
                    st.success("""
                    - Maintain healthy lifestyle
                    - Annual health checkups
                    - Balanced diet with whole foods
                    - Regular physical activity
                    """)

            with col2:
                st.subheader("Key Contributing Factors")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Impact': np.exp(model.coef_[0])
                }).sort_values('Impact', ascending=False)

                st.bar_chart(feature_importance.set_index('Feature')['Impact'])

                st.subheader("Factor Explanations")
                st.write("""
                - **Glucose**: Blood sugar levels (most significant predictor)
                - **BMI**: Body mass index (obesity correlation)
                - **Age**: Risk increases with age
                - **DiabetesPedigreeFunction**: Genetic predisposition
                - **Pregnancies**: Gestational diabetes history
                - **BloodPressure**: Cardiovascular health indicator
                - **SkinThickness**: Body fat distribution
                - **Insulin**: Insulin resistance marker
                """)

            # Model Evaluation Section
            st.markdown("---")
            st.subheader("Model Evaluation & Explainability")

            # Confusion Matrix Plot
            st.write("### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("True Label")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            # ROC Curve Plot
            st.write("### ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()

                   
                
                

