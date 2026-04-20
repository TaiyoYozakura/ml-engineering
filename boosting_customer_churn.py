from altair import Data
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score)



st.set_page_config(page_title="Customer Churn Prediction",layout="wide")
st.title("Customer Churn Prediction Using Boosting")
st.write("Enter customer Details below to Predict Churn: ")

@st.cache_resource
def load_train_model():
    
    # Loading Dataset and Making DataFrame(df)
    
    file_path=r"./Customer-Churn.csv"
    df=pd.read_csv(file_path)

    # Converting Columns to Numeric DataType
    
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors="coerce")

    # Mapping Churn Values from String To Numeric
     
    df['Churn']=df['Churn'].map({"No":0,"Yes":1})

    # Feature and Label Identification and Separation
    x=df.drop(columns=["Churn",'customerID'])
    y=df['Churn']

    # Feature Separation Based on DataTypes
    
    num_fea=x.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_fea=x.select_dtypes(include=["object"]).columns.tolist()

    # Spliting Training and Testing Data for Both Features and Label
    
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=22,stratify=y)

    # Creating Preprocessing Pipelines for Numeric Feature Data
    
    num_transformer=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ])

    # Creating Preprocessing Pipelines for Categorical Feature Data
    
    cat_transformer=Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ])

    # Creating Preprocess Transformer for Columns
    
    preprocessor=ColumnTransformer(
        transformers=[
            ("num",num_transformer,num_fea),
            ("cat",cat_transformer,cat_fea)
            ])

    # Preprocessing the Training Data 
    
    xtrain_processed=preprocessor.fit_transform(xtrain)
    xtest_processed=preprocessor.transform(xtest)

    # Creating a Model Selector using RandomForestClassifier

    selector_model= RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Training the Selector Model from above
    selector_model.fit(xtrain_processed, ytrain)
    selector=SelectFromModel(selector_model, threshold="median", prefit=True)
    
    X_train_selected=selector.transform(xtrain_processed)
    X_test_selected=selector.transform(xtest_processed)
    
    ada_model=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1,random_state=22),
        n_estimators=300,
        learning_rate=0.05,
        random_state=22
        )

    gb_model=GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=22
        )
    
    models={
        "AdaBoost":ada_model,
        "GradientBoost":gb_model
        }

    best_model=None
    best_auc=-1
    results={}

    for name,model in models.items():
        
        model.fit(X_train_selected,ytrain)

        ypred=model.predict(X_test_selected)
        yprob=model.predict_proba(X_test_selected)[:,1]

        acc=accuracy_score(ytest,ypred)
        auc=roc_auc_score(ytest,ypred)
        
        results[name]={
            "model":model,
            "accuracy":acc,
            "auc":auc
        }

        if auc>best_auc:
            best_auc=auc
            best_model=model
            best_model_name=name
    return {
        "preprocessor": preprocessor,
        "selector": selector,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "results": results,
        "numeric_features": num_fea,
        "categorical_features": cat_fea
    }
artifacts= load_train_model()

preprocessor=artifacts["preprocessor"]
selector=artifacts["selector"]
model=artifacts["best_model"]
best_model_name=artifacts["best_model_name"]
results=artifacts["results"]


st.subheader("Model Performance")

col1,col2=st.columns(2)

with col1:
    st.write("AdaBoost")
    st.write(f"Accuracy : {results['AdaBoost']['accuracy']:.4f}")
    st.write(f"ROC-AUC : {results['AdaBoost']['auc']:.4f}")
    
with col2:
    st.write("GradientBoost")
    st.write(f"Accuracy : {results['GradientBoost']['accuracy']:.4f}")
    st.write(f"ROC-AUC : {results['GradientBoost']['auc']:.4f}")
    
st.success(f"Best Model Selected Automatically :{best_model_name}")

st.subheader("Enter Customer Details")
with st.form("prediction_form"):
    col1, col2, col3= st.columns(3)
    with col1:
        gender =st.selectbox("Gender", ["Female", "Male"])
        senior_citizen =st.selectbox("Senior Citizen", [0, 1])
        partner =st.selectbox("Partner", ["Yes", "No"])
        dependents =st.selectbox("Dependents", ["Yes", "No"])
        tenure =st.number_input("Tenure", min_value=0, max_value=100, value=12)
        phone_service =st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines =st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

    with col2:
        internet_service=st.selectbox("Internet Service",["DSL","Fiber Optic","No"])
        online_security=st.selectbox("Online Security",["No Internet Service","No","Yes"])
        online_backup=st.selectbox("Online Backup",["No Internet Service","No","Yes"])
        device_protection=st.selectbox("Device Protection",["No Internet Service","No","Yes"])
        tech_support=st.selectbox("Tech Support",["No Internet Service","No","Yes"])
        streaming_tv=st.selectbox("Streaming TV",["No Internet Service","No","Yes"])
        streaming_movies=st.selectbox("Streaming Movies",["No Internet Service","No","Yes"])

    with col3:
        contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
        paperless_billing=st.selectbox("Paperless Billing",["Yes","No"])
        payment_method=st.selectbox(
            "Payment Method",
            ["Electronic Check",
             "Mailed Check",
             "Bank Transfer(automatic)",
             "Credit Card (automatic)"
             ])
        monthly_charges=st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=8.1)
        total_charges=st.number_input("Total Charges", min_value=0.0, value=858.8, step=8.1)
    
    submitted=st.form_submit_button("Predict Churn")

if submitted:
    input_df=pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior_citizen],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
       "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    })

    # Apply same preprocessing + feature selection
    input_processed = preprocessor.transform(input_df)
    input_selected = selector.transform(input_processed)

    prediction = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {probability:.4f}")

    if prediction == 1:
        st.error("Prediction: Customer is likely to Churn")
    else:
        st.success("Prediction: Customer is likely to Stay")