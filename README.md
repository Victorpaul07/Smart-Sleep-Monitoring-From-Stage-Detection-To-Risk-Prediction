# Smart-Sleep-Monitoring-From-Stage-Detection-To-Risk-Prediction

# Smart Sleep Monitoring System

AI-powered web application for sleep stage detection from PSG/EEG data and lifestyle-based sleep disorder risk prediction. Developed during my Btech CSE(AIML) final year Major Project. Achieves 94% CNN-LSTM sleep staging accuracy and 82.7% hybrid RF-SVM disorder prediction.

## ✨ Features

- Sleep Stage Analysis: CNN-LSTM model processes PSG EDF files → Wake (W), N1, N2, N3, REM detection
- Disorder Risk Prediction: Hybrid RF+SVM VotingClassifier predicts None/Insomnia/Apnea from 11 lifestyle features
- Interactive Dashboard: File upload + form inputs → instant visualizations (confusion matrices, PRF scores, feature importance)
- User Authentication: Registration, login, profile management with SQLite backend
- Production Ready: Model persistence (.h5/.joblib), responsive UI, loading animations

## 📊 Model Performance Highlights

| Model | Accuracy | F1 (Weighted) | Key Strength |
|-------|----------|---------------|--------------|
| CNN-LSTM (Sleep Stages) | 94.7% (train) | - | Temporal EEG patterns |
| Random Forest | 79.3% | 0.790 | Feature importance |
| SVM (RBF) | 81.8% | 0.819 | Margin optimization |
| Hybrid RF+SVM | 82.7% | 0.826 | Ensemble reliability |

Top Risk Factors: Daily Steps (0.28), Physical Activity (0.22), Sleep Quality (0.19), Stress Level (0.17) 

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/smart-sleep-monitor.git
cd smart-sleep-monitor
```

### 2. Environment
```bash
# Python 3.8+
pip install -r requirements.txt
```

### 3. Train Models (Optional - Download pre-trained)
```bash
# Sleep staging (needs Sleep-EDF dataset)
python train_cnn_lstm_sleep_stages.py

# Lifestyle classifier
python train_lifestyle_classifiers.py
```

### 4. Run Web App
```bash
python app.py
```
**Open**: http://localhost:5000

## 📁 Project Structure
```
smart-sleep-monitor/
├── app.py                 # Flask backend + model inference
├── train_cnn_lstm_sleep_stages.py  # CNN-LSTM training
├── train_lifestyle_classifiers.py  # RF/SVM/Hybrid training
├── templates/             # HTML pages (dashboard.html, login.html...)
├── static/styles.css      # Responsive UI
├── models/                # Saved models (.h5, .joblib)
├── uploads/               # PSG files
├── plots/                 # Performance visualizations
└── sleepmonitor.db        # SQLite users DB[file:7]
```

## 🛠️ Model Architecture

### CNN-LSTM (Sleep Staging)
```
Input: EEG 30s epochs (100Hz → 3000 timesteps × 1 channel)
Conv1D(32,7) → MaxPool → Conv1D(64,5) → MaxPool → Conv1D(128,3)
→ LSTM(64) → Dense(64) → Softmax(5 classes: W/N1/N2/N3/REM)
```
**Training**: 84 epochs, 94% accuracy on Sleep-EDF[ file:8] 

### Hybrid RF+SVM (Disorder Prediction)
```
Features (11): Gender, Age, Sleep Hours, Quality, Activity, Stress, BMI, BP, HR, Steps
- SMOTE balancing + StandardScaler
- RF: n_estimators=300, max_depth=10
- SVM: RBF kernel, C=10, gamma=scale
- VotingClassifier(soft) → 82.7% accuracy[file:9]
```

## 🎯 Usage Workflow

1. **Register/Login** → Access dashboard
2. **Upload PSG EDF** (optional) + **Fill lifestyle form**
3. **Analyze** → Get:
   - Sleep stage distribution table
   - Dominant stage highlight
   - Disorder risk + probability
   - Top contributing factors (or healthy habits)
   - Global feature importance ranking
     
## 📈 Key Visualizations Included

```
plots/
├── model_comparison_acc_loss.jpg      # Bar: Acc/Loss by model
├── prf_hybrid_rf_svm.jpg             # Precision/Recall/F1 bars
├── comprehensive_analysis.jpg         # Confusion matrix + feature importance
├── hybrid_sleep_disorder_results.jpg  # Hybrid model metrics
└── LSTM-CNN.jpg                      # Training curve 94%[file:10-18]
```

## 🔗 Datasets Used

- **Sleep-EDF**: PSG/EEG + hypnograms (30s epochs) 
- **SleepHealthLifestyle**: 400+ samples (11 lifestyle features → 3 disorders)
- 
## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask, SQLite |
| ML/DL | TensorFlow 2.15, Scikit-learn, MNE-Python |
| Frontend | HTML5, CSS3, JavaScript, FontAwesome |
| Deployment | Ngrok (local tunnel) |

## ⚙️ Requirements

```txt
flask==2.3.3
tensorflow==2.15.0
scikit-learn==1.3.2
mne==1.6.1
joblib==1.3.2
numpy pandas matplotlib seaborn imbalanced-learn
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

MIT License - free to use in academic/commercial projects.

## 👨‍💻 Author

M.Victor Paul
CSE_AIML
mvictorpaul07@gmail.com | https://www.linkedin.com/in/victorpaulm
