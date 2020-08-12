import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('KNN_model.pkl', 'rb'))

#@app.route('/')
#def home():
#   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    #data = request.get_json(force=True)
    #data = [float (x) for x in request.get_json(force=True)]
    #prediction = model.predict([[np.array(data['latitude','longitude'])]])
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features) 
    data = request.args
    lat = float(request.json['latitude'])
    long = float(request.json['longitude'])
    prediction = model.predict([np.array(lat,long)])
    output = prediction[0]
    return jsonify(output)
    #return render_template('index.html', prediction_text='Activity will be'.format(output))
    
if __name__ == "__main__":
    app.run(debug=True)
