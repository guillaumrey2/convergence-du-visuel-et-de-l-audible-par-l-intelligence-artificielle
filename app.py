from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration des dossiers pour les fichiers uploadés et les fichiers statiques
app.config['UPLOAD_FOLDER'] = '/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def accueil():
    # Affiche la page d'accueil
    return render_template('accueil.html')

@app.route('/loadImage', methods=['GET', 'POST'])
def loadImage():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('recordings', filename=filename))
    return render_template('loadImage.html')

@app.route('/recordings/<filename>')
def recordings(filename):
    audio_file = os.path.join('recordings', filename.rsplit('.', 1)[0] + '.mp3')
    return render_template('recordings.html', filename=filename, audio_file=audio_file)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
