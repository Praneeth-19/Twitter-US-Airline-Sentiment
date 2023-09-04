import numpy as np
import pickle
import pandas as pd
from flask import Flask,request

app=Flask(__name__)

pickle_in = open("cv-transform.pkl",'rb')
classifier=pickle.load(pickle_in)

@app.route('/')    #decorat
def welcome():
    return "Welcome All"

@app.route('/predict',methods=['Get'])
def predict_note_authentication():
    
    input_cols=['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'text', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone']
    
    list1=[]
    for i in input_cols:
        val=request.args.get(i)
        list1.append(val)
    
    prediction=classifier.predict([list1])
    print(prediction)
    return "Hello the answer: "+str(prediction) 


@app.route('/predict_file',methods=['POST'])
def predict_note_file():
    df_test= pd.read_csv(request.files.get('file'))
    prediction=classifier.predict(df_test)
    return str(list(prediction))
        
if __name__== '__main__':
    app.run(host='0.0.0.0',port=8000)