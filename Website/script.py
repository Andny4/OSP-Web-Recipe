from flask import Flask, render_template, redirect, url_for, request
import pickle
import numpy as np
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/',methods=['GET'])
def newpage():
    return render_template('first.html')

@app.route('/AUSTRO-HUNGARIANCREAMOFHORSERADISHSOUP')
def AUSTROHUNGARIANCREAMOFHORSERADISHSOUP():
    return render_template('AUSTRO-HUNGARIANCREAMOFHORSERADISHSOUP.html')

@app.route('/Biryani')
def biryanishorba():
    return render_template('biryanishorba.html')

@app.route('/Vada')
def vada():
    return render_template('vada.html')

@app.route('/Bhindi')
def bhindifry():
    return render_template('bhindifry.html')

@app.route('/Halal')
def hcchicken():
    return render_template('hcchicken.html')

@app.route('/index',methods=['POST'])
def helloworld():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def insurance():
    Vegies=request.form['Vegies']
    Vegies2=request.form['Vegies2']
    Vegies3=request.form['Vegies3']
    Flour_Bread=request.form['Flour/Bread']
    Flour_Bread2=request.form['Flour/Bread2']
    Sauces=request.form['Sauces']
    Sauces2=request.form['Sauces2']
    Sauces3=request.form['Sauces3']
    Meat=request.form['Meat']
    Meat2=request.form['Meat2']
    Rice_Noodles=request.form['Rice/Noodles']
    Spices=request.form['Spices']
    Spices2=request.form['Spices2']
    Spices3=request.form['Spices3']
    Other=request.form['Other']
    Other2=request.form['Other2']
    Diabetic=request.form['Diabetic']
    arr = np.array([[Vegies,Vegies2,Vegies3,Flour_Bread,Flour_Bread2,Sauces,Sauces2,Sauces3,Meat,Meat2,Rice_Noodles,Spices,Spices2,Spices3,Other,Other2,Diabetic]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)

if __name__=='__main__':
    app.run(port=3000,debug=True)

