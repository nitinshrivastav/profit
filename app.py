#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Machine Learning Project Deployement


# In[2]:


import pandas as pd


# In[3]:


data=pd.read_csv("50_Startups.csv")


# In[5]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[6]:


data['State']=le.fit_transform(data['State'])


# In[8]:


x=data.iloc[:,0:-1]
y=data.iloc[:,-1]


# In[9]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


# In[10]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[11]:


model.fit(xtrain,ytrain)


# In[ ]:


from flask import Flask,render_template,request
app=Flask(__name__)
@app.route("/")
def abc():
    return render_template("info.html")
@app.route("/details",methods=['GET','POST'])
def xyz():
    if(request.method=='POST'):
        res=request.form['r1']
        admin=request.form['a1']
        market=request.form['m1']
        state=request.form['s1']
        state=le.transform([state])
        result=model.predict([[res,admin,market,state]])
        return render_template("info.html",answer=result)
if __name__=="__main__":
    app.run()

