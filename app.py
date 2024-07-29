from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Configuration des dossiers pour les fichiers uploadés et les fichiers statiques
app.config['UPLOAD_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/images'
app.config['RECORDINGS_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/recordings'
print("Upload folder set to:", app.config['UPLOAD_FOLDER'])
print("Recordings folder set to:", app.config['RECORDINGS_FOLDER'])
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def accueil():
    return render_template('accueil.html')

@app.route('/loadImage', methods=['GET', 'POST'])
def loadImage():
    logging.debug("loadImage route accessed")
    if request.method == 'POST':
        logging.debug("POST request received at /loadImage")
        if 'file' not in request.files:
            logging.error("No file part in request")
            return redirect(request.url)
        file = request.files['file']
        logging.debug(f"File received: {file}")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logging.debug(f"Saving file to: {path}")
            try:
                file.save(path)
                logging.debug("File saved successfully")
                # Start a background thread to check for the corresponding audio file
                threading.Thread(target=check_for_recording, args=(filename,)).start()
                return render_template('loading.html')
            except Exception as e:
                logging.error(f"Error saving file: {e}")
                return "Error saving file", 500
        else:
            logging.error("File not allowed or not received")
    return render_template('loadImage.html')

def check_for_recording(image_filename):
    base_filename = os.path.splitext(image_filename)[0]
    expected_audio_file = base_filename + '.wav'
    while True:
        time.sleep(5)  # Check every 5 seconds
        if os.path.isfile(os.path.join(app.config['RECORDINGS_FOLDER'], expected_audio_file)):
            # Redirect to the recordings page
            logging.debug(f"Recording found: {expected_audio_file}")
            with app.test_request_context():
                redirect_url = url_for('recordings', filename=image_filename)
                logging.debug(f"Redirecting to: {redirect_url}")
                return redirect(redirect_url)
        else:
            logging.debug("Recording not found yet...")

@app.route('/recordings/<filename>')
def recordings(filename):
    base_filename = os.path.splitext(filename)[0]
    audio_file = os.path.join(app.config['RECORDINGS_FOLDER'], base_filename + '.wav')
    if os.path.isfile(audio_file):
        return render_template('recordings.html', filename=filename, audio_file=audio_file)
    else:
        return "Recording not found", 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
