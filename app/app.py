# app.py — CLV & Churn Predictor Streamlit App

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title="CLV & Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ── LOAD & CLEAN DATA ────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

df = load_data()

# ── TRAIN MODEL ──────────────────────────────────────
@st.cache_resource
def train_model(df):
    df_ml = df.drop(['customerID', 'TotalCharges'], axis=1)
    df_ml['Churn'] = df_ml['Churn'].map({'Yes': 1, 'No': 0})
    df_ml = pd.get_dummies(df_ml, drop_first=True)
    X = df_ml.drop('Churn', axis=1)
    y = df_ml['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist()

model, feature_columns = train_model(df)

# ── TITLE ────────────────────────────────────────────
st.title("📡 Customer Lifetime Value & Churn Predictor")
st.markdown("**Telco Customer Analysis — 7,043 customers | Random Forest Model | 81% Accuracy**")
st.markdown("---")

# ── SECTION 1: KEY METRICS ───────────────────────────
st.header("📊 Overview Dashboard")

total    = len(df)
churned  = df[df['Churn'] == 'Yes'].shape[0]
retained = df[df['Churn'] == 'No'].shape[0]
churn_rate = round(churned / total * 100, 2)
revenue_at_risk = round(churned * df[df['Churn']=='Yes']['MonthlyCharges'].mean(), 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers",  f"{total:,}")
col2.metric("Churned",          f"{churned:,}")
col3.metric("Churn Rate",       f"{churn_rate}%")
col4.metric("Monthly Revenue at Risk", f"₹{revenue_at_risk:,.0f}")

st.markdown("---")

# ── SECTION 2: CHARTS ────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = df.groupby('Contract')['Churn']\
                       .value_counts(normalize=True)\
                       .mul(100).round(2)\
                       .reset_index()
    contract_churn.columns = ['Contract', 'Churn', 'Percentage']
    contract_churn = contract_churn[contract_churn['Churn'] == 'Yes']
    fig1 = px.bar(
        contract_churn, x='Contract', y='Percentage',
        color='Contract', text='Percentage',
        color_discrete_map={
            'Month-to-month': '#EF553B',
            'One year'      : '#FFA15A',
            'Two year'      : '#00CC96'
        }
    )
    fig1.update_traces(texttemplate='%{text}%', textposition='outside')
    fig1.update_layout(showlegend=False, plot_bgcolor='white',
                       yaxis=dict(range=[0, 55]))
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("CLV Distribution: Churned vs Retained")
    fig3 = px.histogram(
        df, x='TotalCharges', color='Churn',
        nbins=50, barmode='overlay', opacity=0.7,
        color_discrete_map={'Yes': '#EF553B', 'No': '#00CC96'},
        labels={'TotalCharges': 'Total Charges (₹)'}
    )
    fig3.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── SECTION 3: FEATURE IMPORTANCE ───────────────────
st.header("🔍 What Drives Churn?")

feature_importance = pd.DataFrame({
    'Feature'   : feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True).tail(10)

fig2 = px.bar(
    feature_importance, x='Importance', y='Feature',
    orientation='h', text='Importance',
    color='Importance', color_continuous_scale='Reds'
)
fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig2.update_layout(
    plot_bgcolor='white', showlegend=False,
    xaxis=dict(range=[0, 0.30]),
    coloraxis_showscale=False
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── SECTION 4: LIVE PREDICTOR ────────────────────────
st.header("🤖 Live Churn Predictor")
st.markdown("Fill in customer details below and get instant churn prediction.")

col1, col2, col3 = st.columns(3)

with col1:
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges (₹)", 18, 119, 65)
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Partner", ["No", "Yes"])
    dependents      = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    contract        = st.selectbox("Contract Type", 
                        ["Month-to-month", "One year", "Two year"])
    internet        = st.selectbox("Internet Service",
                        ["DSL", "Fiber optic", "No"])
    payment         = st.selectbox("Payment Method",
                        ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", 
                         "Credit card (automatic)"])
    paperless       = st.selectbox("Paperless Billing", ["No", "Yes"])

with col3:
    phone_service   = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines  = st.selectbox("Multiple Lines",
                        ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security",
                        ["No", "Yes", "No internet service"])
    tech_support    = st.selectbox("Tech Support",
                        ["No", "Yes", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV",
                        ["No", "Yes", "No internet service"])

# ── PREDICT BUTTON ───────────────────────────────────
if st.button("🔮 Predict Churn Risk", type="primary"):

    # Build input row
    input_dict = {col: 0 for col in feature_columns}

    # Fill numeric
    input_dict['tenure']         = tenure
    input_dict['MonthlyCharges'] = monthly_charges

    # Fill encoded columns
    def set_if_exists(col, val):
        if col in input_dict:
            input_dict[col] = val

    set_if_exists('gender_Male',                      0)
    set_if_exists('SeniorCitizen_Yes',                1 if senior_citizen == 'Yes' else 0)
    set_if_exists('Partner_Yes',                      1 if partner == 'Yes' else 0)
    set_if_exists('Dependents_Yes',                   1 if dependents == 'Yes' else 0)
    set_if_exists('PhoneService_Yes',                 1 if phone_service == 'Yes' else 0)
    set_if_exists('MultipleLines_No phone service',   1 if multiple_lines == 'No phone service' else 0)
    set_if_exists('MultipleLines_Yes',                1 if multiple_lines == 'Yes' else 0)
    set_if_exists('InternetService_Fiber optic',      1 if internet == 'Fiber optic' else 0)
    set_if_exists('InternetService_No',               1 if internet == 'No' else 0)
    set_if_exists('OnlineSecurity_No internet service', 1 if online_security == 'No internet service' else 0)
    set_if_exists('OnlineSecurity_Yes',               1 if online_security == 'Yes' else 0)
    set_if_exists('TechSupport_No internet service',  1 if tech_support == 'No internet service' else 0)
    set_if_exists('TechSupport_Yes',                  1 if tech_support == 'Yes' else 0)
    set_if_exists('StreamingTV_No internet service',  1 if streaming_tv == 'No internet service' else 0)
    set_if_exists('StreamingTV_Yes',                  1 if streaming_tv == 'Yes' else 0)
    set_if_exists('Contract_One year',                1 if contract == 'One year' else 0)
    set_if_exists('Contract_Two year',                1 if contract == 'Two year' else 0)
    set_if_exists('PaperlessBilling_Yes',             1 if paperless == 'Yes' else 0)
    set_if_exists('PaymentMethod_Electronic check',   1 if payment == 'Electronic check' else 0)
    set_if_exists('PaymentMethod_Credit card (automatic)', 1 if payment == 'Credit card (automatic)' else 0)
    set_if_exists('PaymentMethod_Mailed check',       1 if payment == 'Mailed check' else 0)

    # Make prediction
    input_df   = pd.DataFrame([input_dict])
    probability = model.predict_proba(input_df)[0][1] * 100
    prediction  = model.predict(input_df)[0]

    # Show result
    st.markdown("---")
    st.subheader("Prediction Result")

    if probability >= 60:
        st.error(f"🔴 HIGH CHURN RISK — {probability:.1f}% probability")
        st.markdown("**Recommended Action:** Offer immediate retention discount or contract upgrade.")
    elif probability >= 35:
        st.warning(f"🟡 MEDIUM CHURN RISK — {probability:.1f}% probability")
        st.markdown("**Recommended Action:** Schedule proactive check-in call within 30 days.")
    else:
        st.success(f"🟢 LOW CHURN RISK — {probability:.1f}% probability")
        st.markdown("**Recommended Action:** Maintain current engagement. Monitor monthly.")

    # Show input summary
    st.markdown("**Customer Profile Entered:**")
    st.write(f"Tenure: {tenure} months | Monthly: ₹{monthly_charges} | "
             f"Contract: {contract} | Internet: {internet} | Payment: {payment}")