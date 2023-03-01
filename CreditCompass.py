# Imports and Dependencies

import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle as pkl
import shap
import streamlit.components.v1 as components

# Load the saved model

model=pkl.load(open("input/model.p","rb"))

# Browser-Tab Configuration

st.set_page_config(
    page_title="Credit Compass",
    page_icon="images/compass.png"
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Main-Page

st.title("Welcome to Credit Compass :compass:")
st.subheader("To get started, enter your details on the left-hand-side panel and hit GO! Credit Compass will handle the rest :rocket:")

st.markdown("<br>", unsafe_allow_html = True)

st.write("""Credit Compass uses advanced machine learning algorithms that have been trained on the lending history of millions 
    of applicants, meaning it knows exactly what it takes to get approved for finance.""")

st.markdown("<br>", unsafe_allow_html = True)

col1, col2 = st.columns([1, 1])

with col1:
    st.write("""Unlike a human, it is available to provide 
    insights into the liklihood of your credit approval 24 hours a day. Enjoy fast answers in the comfort of your own home while leaving 
    your credit rating undisturbed. You'll never have to worry about being in the dark regarding your creditworthiness ever again.""")

with col2:
    st.image("images/easyPeasy.png")

st.markdown("<br>", unsafe_allow_html = True)

col3, col4 = st.columns([1, 1])

with col3:
    st.image("images/lock.jpg")

with col4:
    st.write("""Credit Compass's unique algorithm is completely contained and does not access any of your records 
    or your credit file, so you can rest assured that your credit score will remain untouched. Enjoy transparent financial advice and an 
    expedited approval process, on us.""")

st.markdown("<br>", unsafe_allow_html = True)

st.write("""<p style='font-size: 14px'>Gross earnings may be input between a range of $10,000 to $200,000 per year. Loans are 
         available from $1,000 to $40,000, and for terms of 36 or 60 months. Please contact your local lending specialist if you require 
         any assistance with the listed entry fields, or entries (income, loan term etc.) would lie outside of the fields 
         that have been provided.</p>""", unsafe_allow_html = True)
st.write("""<p style='font-size: 10px'>Disclaimer: Credit Compass is only an indication of whether you may be suitable to receive credit, 
         however there may be other factors influencing the decision. The information provided does not therefore constitute financial 
         advice and Credit Compass or its subsidiaries are not liable for any decisions made by you as a result of receiving the prediction below. By selecting 
         'GO!' and using our calculation tool, you accept these conditions of use.</p>""", unsafe_allow_html = True)

st.subheader("You'll find your creditworthiness indicated below!")

# The Sidebar

st.sidebar.image("images/compass.png", width = 285)
st.sidebar.title("Enter your details here :point_down:")

# Sidebar Input features

## Annual Income
annual_inc =st.sidebar.slider("What is your current gross annual earnings (before tax)?", min_value=10000, max_value=200000,step=1000)

## Employment Length
emp_length = st.sidebar.selectbox("For how long have you been employed at your current job?", ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                                                                           "6 years", "7 years","8 years","9 years","10+ years") )

## Loan Amount
loan_amnt =st.sidebar.slider("How much are you looking to borrow?",min_value=1000, max_value=40000,step=500)

## Term
term = st.sidebar.radio("For what length of time do you require a loan?", ('36 months', '60 months'))

## Sub Grade
sub_grade =st.sidebar.selectbox('What is your current credit grade? If you are unsure or require assistance, our lending specialists are available to assist. If you believe you have great credit and have always paid on time and in full, simply leave this as A1', 
                                                        ("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3","B4","B5","C1", "C2", "C3",
                                                        "C4", "C5", "D1", "D2", "D3","D4","D5", "E1", "E2", "E3","E4","E5","F1", "F2",
                                                        "F3", "F4", "F5", "G1", "G2", "G3","G4","G5"))

## DTI
dti=st.sidebar.slider("What is your current DTI? If you do not have any loans, please leave as 0.10. You may also easily estimate this yourself by dividing your debt obligations excluding mortgage by your gross monthly income",min_value=0.1, max_value=100.1,step=0.1)

## Time since previous enquiry
mths_since_recent_inq=st.sidebar.slider("How long has it been since you last enquired about or applied for finance? Please indicate below in months of time passed",min_value=1, max_value=25,step=1)

## Revolving Credit Utilised
revol_util=st.sidebar.slider("Revolving Credit is the amount of credit you have available vs. the amount of credit you've utilised. For example, if you have all your credit available and no credit-drawn, enter 100",min_value=0.1, max_value=150.1,step=0.1)

## Number of Open Revolving Accounts
num_op_rev_tl=st.sidebar.slider("Following on from above, please indicate the amount of revolving accounts that you have open, or in other words, how many credit accounts do you currently have?",min_value=1, max_value=50,step=1)

def preprocess(loan_amnt, term, sub_grade, emp_length, annual_inc, dti, mths_since_recent_inq, revol_util, num_op_rev_tl):
    # Pre-processing user input

    user_input_dict={'loan_amnt':[loan_amnt], 'term':[term], 'sub_grade':[sub_grade], 'emp_length':[emp_length], 'annual_inc':[annual_inc], 'dti':[dti],
                'mths_since_recent_inq':[mths_since_recent_inq], 'revol_util':[revol_util], 'num_op_rev_tl':[num_op_rev_tl]}
    user_input=pd.DataFrame(data=user_input_dict)

    cleaner_type = {"term": {"36 months": 1.0, "60 months": 2.0},
    "sub_grade": {"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0, "A5": 5.0,
    "B1": 11.0, "B2": 12.0, "B3": 13.0, "B4": 14.0, "B5": 15.0,
    "C1": 21.0, "C2": 22.0, "C3": 23.0, "C4": 24.0, "C5": 25.0,
    "D1": 31.0, "D2": 32.0, "D3": 33.0, "D4": 34.0, "D5": 35.0,
    "E1": 41.0, "E2": 42.0, "E3": 43.0, "E4": 44.0, "E5": 45.0,
    "F1": 51.0, "F2": 52.0, "F3": 53.0, "F4": 54.0, "F5": 55.0,
    "G1": 61.0, "G2": 62.0, "G3": 63.0, "G4": 64.0, "G5": 65.0,
    },
    "emp_length": {"< 1 year": 0.0, '1 year': 1.0, '2 years': 2.0, '3 years': 3.0, '4 years': 4.0,
    '5 years': 5.0, '6 years': 6.0, '7 years': 7.0, '8 years': 8.0, '9 years': 9.0,
    '10+ years': 10.0 }
    }

    user_input = user_input.replace(cleaner_type)

    return user_input

user_input=preprocess(loan_amnt, term, sub_grade, emp_length, annual_inc, dti, mths_since_recent_inq, revol_util, num_op_rev_tl)

# The Predict Button!

btn_predict = st.sidebar.button("GO!")

if btn_predict:
    pred = model.predict_proba(user_input)[:, 1]

    if pred[0] < 0.78:
        st.error("""Thank you for your interest in Credit Compass. Unfortunately, it doesn't look like we could approve you 
        for credit at this time, however please do try again when things are a bit different. While we may not be the right fit for eachother 
        now, we could certainly be in the future. Hope to see you then!""")
    else:
        st.success("""You are likely to receive credit! Please feel free to reach out to our branch representative for a 
        quick chat and streamlined credit approval process. Thank you for using Credit Compass!""")

    #prepare test set for shap explainability
    loans = st.cache(pd.read_csv)("input/mycsvfile.csv.gz")
    X = loans.drop(columns=['loan_status','home_ownership__ANY','home_ownership__MORTGAGE','home_ownership__NONE','home_ownership__OTHER','home_ownership__OWN',
                   'home_ownership__RENT','addr_state__AK','addr_state__AL','addr_state__AR','addr_state__AZ','addr_state__CA','addr_state__CO','addr_state__CT',
                   'addr_state__DC','addr_state__DE','addr_state__FL','addr_state__GA','addr_state__HI','addr_state__ID','addr_state__IL','addr_state__IN',
                   'addr_state__KS','addr_state__KY','addr_state__LA','addr_state__MA','addr_state__MD','addr_state__ME','addr_state__MI','addr_state__MN',
                   'addr_state__MO','addr_state__MS','addr_state__MT','addr_state__NC','addr_state__ND','addr_state__NE','addr_state__NH','addr_state__NJ',
                   'addr_state__NM','addr_state__NV','addr_state__NY','addr_state__OH','addr_state__OK','addr_state__OR','addr_state__PA','addr_state__RI',
                   'addr_state__SC','addr_state__SD','addr_state__TN','addr_state__TX','addr_state__UT','addr_state__VA','addr_state__VT', 'addr_state__WA',
                   'addr_state__WI','addr_state__WV','addr_state__WY'])
    y = loans[['loan_status']]
    y_ravel = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y_ravel, test_size=0.25, random_state=42, stratify=y)

    st.subheader('Result Interpretability - Applicant Level')
    shap.initjs()
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(user_input)
    fig = shap.plots.bar(shap_values[0])
    st.pyplot(fig)
    st.write("""#175_pls help interprete the plot""")


    st.subheader('Model Interpretability - Overall')
    shap_values_ttl = explainer(X_test)
    fig_ttl = shap.plots.beeswarm(shap_values_ttl)
    st.pyplot(fig_ttl)
    st.write(""" This beeswarm plot shows the SHAP values for each feature in the test set (X_test).
    """)
