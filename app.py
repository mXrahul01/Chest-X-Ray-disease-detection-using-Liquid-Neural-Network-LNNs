from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import csv

app = Flask(__name__)

# Use absolute path for upload folder
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ---------------- Liquid Neural Net Classes ---------------- #
class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidNeuron, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h_prev):
        h_current = (1 - self.tau) * h_prev + self.tau * torch.tanh(self.W_in(x) + self.W_rec(h_prev))
        return h_current

class LiquidNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNN, self).__init__()
        self.liquid_neuron = LiquidNeuron(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.liquid_neuron.hidden_size).to(x.device)
        x = x.view(batch_size, -1)  # Flatten image
        h = self.liquid_neuron(x, h)
        output = self.fc(h)
        return output

# -------------------- Model Loading -------------------- #
# Load models (CPU for deployment here)
model1 = LiquidNN(input_size=224*224, hidden_size=128, output_size=2)
model2 = LiquidNN(input_size=224*224, hidden_size=128, output_size=2)
model1.load_state_dict(torch.load(
    r'C:\Users\rahul\WebstormProjects\Chest-X-Ray-disease-detection-using-Liquid Neural Network(LNNs)\models\liquid_model.pth',
    map_location=torch.device('cpu')))
model2.load_state_dict(torch.load(
    r'C:\Users\rahul\WebstormProjects\Chest-X-Ray-disease-detection-using-Liquid Neural Network(LNNs)\models\best_model.pth',
    map_location=torch.device('cpu')))
model1.eval()
model2.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------- ROUTES ------------------- #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'folder' not in request.files:
            return redirect(request.url)
        folder = request.files.getlist('folder')
        results = []
        for file in folder:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                try:
                    image = Image.open(file_path).convert('L')
                    image = transform(image)
                    image = image.unsqueeze(0)
                    with torch.no_grad():
                        output1 = model1(image)
                        prob1 = torch.softmax(output1, dim=1)
                        confidence1, pred1 = torch.max(prob1, 1)
                        output2 = model2(image)
                        prob2 = torch.softmax(output2, dim=1)
                        confidence2, pred2 = torch.max(prob2, 1)
                    labels1 = ['Normal', 'Pneumonia']
                    labels2 = ['Normal', 'Lung Opacity']
                    prediction1 = labels1[pred1.item()]
                    confidence1 = confidence1.item()
                    prediction2 = labels2[pred2.item()]
                    confidence2 = confidence2.item()
                    if confidence1 > confidence2:
                        final_prediction = f"{prediction1} (Confidence: {confidence1:.2f})"
                        final_confidence = confidence1
                        is_disease = prediction1 != 'Normal'
                    else:
                        final_prediction = f"{prediction2} (Confidence: {confidence2:.2f})"
                        final_confidence = confidence2
                        is_disease = prediction2 != 'Normal'
                    results.append({
                        'filename': filename,
                        'prediction1': prediction1,
                        'confidence1': confidence1,
                        'prediction2': prediction2,
                        'confidence2': confidence2,
                        'final_prediction': final_prediction,
                        'final_confidence': final_confidence,
                        'is_disease': is_disease
                    })
                finally:
                    # DELETE the uploaded image after prediction to save storage
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {e}")

        disease_results = [r for r in results if r['is_disease']]
        normal_results = [r for r in results if not r['is_disease']]
        disease_results.sort(key=lambda x: -x['final_confidence'])

        # Save all results to results.csv (generic download)
        results_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
        with open(results_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Model 1 Prediction', 'Model 1 Confidence',
                             'Model 2 Prediction', 'Model 2 Confidence', 'Final Prediction', 'Is Disease'])
            for result in results:
                writer.writerow([
                    result['filename'], result['prediction1'], result['confidence1'],
                    result['prediction2'], result['confidence2'],
                    result['final_prediction'], result['is_disease']
                ])

        # Save separate CSVs for disease/normal
        disease_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results_disease.csv')
        with open(disease_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Model 1 Prediction', 'Model 1 Confidence',
                             'Model 2 Prediction', 'Model 2 Confidence', 'Final Prediction'])
            for result in disease_results:
                writer.writerow([
                    result['filename'], result['prediction1'], result['confidence1'],
                    result['prediction2'], result['confidence2'],
                    result['final_prediction']
                ])

        normal_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results_normal.csv')
        with open(normal_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Model 1 Prediction', 'Model 1 Confidence',
                             'Model 2 Prediction', 'Model 2 Confidence', 'Final Prediction'])
            for result in normal_results:
                writer.writerow([
                    result['filename'], result['prediction1'], result['confidence1'],
                    result['prediction2'], result['confidence2'],
                    result['final_prediction']
                ])
        return redirect(url_for('loading'))
    return render_template('upload.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/result')
def result():
    disease_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results_disease.csv')
    disease_results = []
    if os.path.exists(disease_csv_path):
        with open(disease_csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                disease_results.append({
                    'filename': row[0],
                    'prediction1': row[1],
                    'confidence1': float(row[2]),
                    'prediction2': row[3],
                    'confidence2': float(row[4]),
                    'final_prediction': row[5]
                })
    disease_results.sort(key=lambda x: max(x['confidence1'], x['confidence2']), reverse=True)
    top_disease_results = disease_results[:15]
    return render_template('result.html', results=top_disease_results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
