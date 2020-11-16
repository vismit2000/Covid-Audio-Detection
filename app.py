from flask import Flask, render_template, url_for, request, redirect
from predict_class import *
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def hello():
	return render_template('index.html')

@app.route('/', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		audio_file = request.files['audio']
		audio_file.save("static/"+audio_file.filename)

		audio_type = 'cough'
		is_cough_symptom = 1
	
		prediction = predict("static/"+audio_file.filename,  audio_type, is_cough_symptom)
		
	return render_template('index.html', results = prediction)

if __name__ == '__main__':
	app.run(debug = True)