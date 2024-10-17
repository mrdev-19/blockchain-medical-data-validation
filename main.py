import streamlit as st
from streamlit_option_menu import option_menu
import database as db
import validations as val
import predict as pr
import chain as ch
import time
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#---------------------------------------------------
# page config settings:

page_title="Insurance Claim Validator"
page_icon=":hospital:"
layout="centered"

st.set_page_config(page_title=page_title,page_icon=page_icon,layout=layout)
st.title(page_title+" "+page_icon)

#--------------------------------------------------
#hide the header and footer     

hide_ele="""
        <style>
        #Mainmenu {visibility:hidden;}
        footer {visibility:hidden;} 
        header {visibility:hidden;}
        </style>
        """
st.markdown(hide_ele,unsafe_allow_html=True)
#---------------------------------------------------
curlogin=""
def log_sign():
    selected=option_menu(
        menu_title=None,
        options=["Login","Signup"],
        icons=["bi bi-link-45deg","bi bi-arrow-bar-up "],
        orientation="horizontal"
    )
    global submit
    if(selected=="Login"):
        with st.form("Login",clear_on_submit=True):
            st.header("Login")
            username=st.text_input("Username")
            password=st.text_input("Password",type="password")
            submit=st.form_submit_button()
            if(submit):
                if(username=="" or password==""):
                    st.warning("Enter your login credentials")
                else:
                    if(db.authenticate(username,password)):
                        st.session_state["curlogin"]=username
                        st.session_state["key"]="main"
                        st.experimental_rerun()
                    else:
                        st.error("Please check your username / password ")
            
    elif(selected=="Signup"):
         with st.form("Sign Up",clear_on_submit=False):
            st.header("Sign Up")
            email=st.text_input("Enter your email")
            number=st.text_input("Enter your Mobile Number")
            username=st.text_input("Enter your username")
            password=st.text_input("Enter your password",type="password")
            submit=st.form_submit_button()
            if(submit):
                dev=db.fetch_all_users()
                usernames=[]
                emails=[]
                numbers=[]
                for user in dev:
                    usernames.append(user["key"])
                    emails.append(user["email"])
                    numbers.append(user["number"])
                var=True
                if(val.validate_email(email)==False):
                    st.error("Enter email in a valid format like 'yourname@org.com'")
                elif(email in emails):
                    st.error("email already exists!\nTry with another email !")
                elif(val.validate_mobile(number)==False):
                    st.error("Please Check your mobile Number")
                elif(number in numbers):
                    st.error("Phone number already exists\nTry with another number")
                elif(val.validate_username(username)==False):
                    st.error("Invalid Username!\nUsername must be between 4-20 characters and can contain only _ and . , and username cannot begin with special characters")
                elif(username in usernames):
                    st.error("Username already exists!\nTry another username !")
                elif(val.validate_password(password)==False):
                    st.error("Password must be between 6-20 characters in length and must have at least one Uppercase Letter , Lowercase letter , numeric character and A Special Symbol(#,@,$,%,^,&,+,=)")
                elif(var):
                    db.insert_user(username,password,email,number)
                    st.success("Signed Up Successfully....Redirecting!!")
                    time.sleep(2)
                    st.session_state["curlogin"]=username
                    st.session_state["key"]="main"
                    st.experimental_rerun()
    
    elif selected=="Admin":
        with st.form("Admin Login",clear_on_submit=True):
            st.header("Admin Login")
            username=st.text_input("Username")
            password=st.text_input("Password",type="password")
            submit=st.form_submit_button()
            if(submit):
                if(username=="" or password==""):
                    st.warning("Enter your login credentials")
                else:
                    if(db.ad_authenticate(username,password)):
                        st.session_state["curlogin"]=username
                        st.session_state["key"]="adminmain"
                        st.experimental_rerun()
                    else:
                        st.error("Please check your username / password ")



def main():
    selected = option_menu(
        menu_title=None,
        options=["Claim Validator","Blockchain"],
        icons=["bi bi-search"],
        orientation="horizontal"
    )
    if selected == "Claim Validator":
        with st.form("Claim_val", clear_on_submit=True):
            st.header("Claims Validation")
            uploaded_file = st.file_uploader("Choose a file")
            submitted = st.form_submit_button("Submit data")
            if submitted:
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of the Uploaded Data:")
                    st.write(df.head())
                    dev=pr.pred(df)
                    st.write(dev)

    elif selected=="Blockchain":
        df = pd.read_csv("output.csv")
        st.write(df)

if "key" not in st.session_state:
    st.session_state["key"] = "log_sign"

if st.session_state["key"] == "log_sign":
    log_sign()

elif st.session_state["key"] == "main":
    main()