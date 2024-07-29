from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/images'
app.config['RECORDINGS_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/recordings'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def accueil():
    return render_template('accueil.html')

@app.route('/loadImage', methods=['GET', 'POST'])
def loadImage():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            return redirect(url_for('loading', filename=filename))
    return render_template('loadImage.html')

@app.route('/loading/<filename>')
def loading(filename):
    return render_template('loading.html', filename=filename)

@app.route('/check_status/<filename>')
def check_status(filename):
    expected_audio_file = os.path.splitext(filename)[0] + '.wav'
    if os.path.isfile(os.path.join(app.config['RECORDINGS_FOLDER'], expected_audio_file)):
        return jsonify({'ready': True})
    return jsonify({'ready': False})

@app.route('/recordings/<filename>')
def recordings(filename):
    audio_file = os.path.join(app.config['RECORDINGS_FOLDER'], os.path.splitext(filename)[0] + '.wav')
    if os.path.isfile(audio_file):
        return render_template('recordings.html', filename=filename, audio_file=audio_file)
    return "Recording not found", 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
