from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/images'
app.config['RECORDINGS_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/recordings'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def accueil():
    return render_template('accueil.html')

@app.route('/loadImage', methods=['GET', 'POST'])
def loadImage():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                logging.debug("File saved successfully")
                return render_template('loading.html', filename=filename)  # Pass filename to the template
            except Exception as e:
                logging.error(f"Error saving file: {e}")
                return "Error saving file", 500
    return render_template('loadImage.html')

@app.route('/check_status/<filename>')
def check_status(filename):
    audio_file = os.path.join(app.config['RECORDINGS_FOLDER'], filename.rsplit('.', 1)[0] + '.wav')
    if os.path.isfile(audio_file):
        return jsonify({'ready': True})
    return jsonify({'ready': False})

@app.route('/recordings/<filename>')
def recordings(filename):
    base_filename = os.path.splitext(filename)[0]
    audio_filename = base_filename + '.wav'
    audio_file = os.path.join(app.config['RECORDINGS_FOLDER'], audio_filename)
    if os.path.isfile(audio_file):
        # Pass the filename for downloading and the audio filename for playing
        return render_template('recordings.html', filename=filename, audio_filename=audio_filename)
    return "Recording not found", 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
