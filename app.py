# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the Flask application
app = Flask(__name__)

# Configure upload and recordings folders and allowed extensions
app.config['UPLOAD_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/images'
app.config['RECORDINGS_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/recordings'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if a file is allowed based on its extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for the home page
@app.route('/')
def accueil():
    return render_template('accueil.html')

# Route to load an image
@app.route('/loadImage', methods=['GET', 'POST'])
def loadImage():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logging.debug("File saved successfully")
            return redirect(url_for('loading', filename=filename))
    return render_template('loadImage.html')

# Route to show the loading page
@app.route('/loading/<filename>')
def loading(filename):
    return render_template('loading.html', filename=filename)

# Route to check the status of a file
@app.route('/check_status/<filename>')
def check_status(filename):
    done_flag_path = os.path.join(app.config['RECORDINGS_FOLDER'], filename.rsplit('.', 1)[0] + '.wav.done')
    if os.path.isfile(done_flag_path):
        return jsonify({'ready': True})
    return jsonify({'ready': False})

# Route to display the recordings page
@app.route('/recordings/<filename>')
def recordings(filename):
    audio_filename = os.path.splitext(filename)[0] + '.wav'
    audio_file = os.path.join(app.config['RECORDINGS_FOLDER'], audio_filename)
    if os.path.isfile(audio_file):
        return render_template('recordings.html', filename=filename, audio_filename=audio_filename)
    return "Recording not found", 404

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        folder = app.config['UPLOAD_FOLDER']
    else:
        folder = app.config['RECORDINGS_FOLDER']
    
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        if filename.lower().endswith(('.wav', '.mp3')):
            mime_type = 'audio/wav'
        else:
            mime_type = 'image/jpeg'
        return send_file(file_path, mimetype=mime_type)
    else:
        app.logger.error("File not found: " + file_path)
        return "File not found", 404

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
