from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import logging

# Initialize Flask and logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Configuration constants
app.config['UPLOAD_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/images'
app.config['RECORDINGS_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/recordings'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Helper function to check file extensions
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
            file.save(filepath)
            logging.debug("File saved successfully")
            return redirect(url_for('loading', filename=filename))
    return render_template('loadImage.html')

@app.route('/loading/<filename>')
def loading(filename):
    return render_template('loading.html', filename=filename)

@app.route('/check_status/<filename>')
def check_status(filename):
    done_flag_path = os.path.join(app.config['RECORDINGS_FOLDER'], filename.rsplit('.', 1)[0] + '.wav.done')
    if os.path.isfile(done_flag_path):
        return jsonify({'ready': True})
    return jsonify({'ready': False})

@app.route('/recordings/<filename>')
def recordings(filename):
    audio_filename = os.path.splitext(filename)[0] + '.wav'
    audio_file = os.path.join(app.config['RECORDINGS_FOLDER'], audio_filename)
    if os.path.isfile(audio_file):
        return render_template('recordings.html', filename=filename, audio_filename=audio_filename)
    return "Recording not found", 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['RECORDINGS_FOLDER'], filename)
    if os.path.exists(file_path):
        try:
            return send_file(file_path, mimetype='audio/wav')
        except Exception as e:
            app.logger.error(f"Failed to send file: {e}")
            return "Error sending file", 500
    else:
        app.logger.error("File not found: " + file_path)
        return "File not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
