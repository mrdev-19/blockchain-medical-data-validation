import os
from deta import Deta
from dotenv import load_dotenv
import pickle
#load env var

load_dotenv(".env")

DETA_KEY="d0hjgdfjri1_23thC9qXWjjLaFqs2W5nVx6yxT9XGB7w"
deta=Deta(DETA_KEY)


ldb=deta.Base("DataHash")
rdb=deta.Base("Data")
cred=deta.Base("Creds")

def insert_user(username,password,email,number):
    cred.put({"key":username,"password":password,"email":email,"number":number})

def insert_admin(username,password,email,number):
    admin.put({"key":username,"password":password,"email":email,"number":number})

def authenticate(username,password):
    var=1
    dev=fetch_all_users()
    usernames=[user["key"] for user in dev]
    emails=[user["email"] for user in dev]
    for user in dev:
        if(username==user["key"] and user["password"]==password):
            return True
            var=0
    if(var):
        return False

def fetch_all_users():
    res=cred.fetch()
    return res.items