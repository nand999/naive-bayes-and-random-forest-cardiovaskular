import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QComboBox, QMessageBox
from PyQt5.QtGui import QIcon, QFont, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def resource_path(relative_path):
    """ Get the absolute path to a resource, works for both development and PyInstaller bundling """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

##membaca file csv
df = pd.read_csv(resource_path('cardio_train.csv'), sep=',')
df['age'] = (df['age'] / 365).astype(int)  
df.drop_duplicates(inplace=True)
df['gender'] = df['gender'].replace({1: 'Male', 2: 'Female'}).replace({'Male': 1, 'Female': 2})
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)  


features = df.drop(columns=['id', 'cardio'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


SE = SMOTEENN()
x_se, y_se = SE.fit_resample(scaled_features, df['cardio'])


x_train, x_test, y_train, y_test = train_test_split(x_se, y_se, test_size=0.20, random_state=0)


nb = GaussianNB()
nb.fit(x_train, y_train)

rf = RandomForestClassifier(random_state=0)
rf.fit(x_train, y_train)

##prediksi naive bayes dan random forest
nb_pred = nb.predict(x_test)
rf_pred = rf.predict(x_test)

##tes akurasi naive bayes
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_class_report = classification_report(y_test, nb_pred)
##tes akurasi random forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_class_report = classification_report(y_test, rf_pred)


class CardiovascularApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Deteksi Risiko Kardiovaskular')
        self.setWindowIcon(QIcon(resource_path('heart_icon.png')))
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()

        header_font = QFont('Arial', 12, QFont.Bold)
        
        self.age_label = QLabel('Usia:')
        self.age_label.setFont(header_font)
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Masukkan usia")
        self.age_input.setValidator(QIntValidator(0, 120))
        self.layout.addWidget(self.age_label)
        self.layout.addWidget(self.age_input)


        self.gender_label = QLabel('Gender (1 = Laki-laki, 2 = Perempuan):')
        self.gender_label.setFont(header_font)
        self.gender_input = QComboBox()
        self.gender_input.addItems(['1', '2'])
        self.layout.addWidget(self.gender_label)
        self.layout.addWidget(self.gender_input)

        self.height_label = QLabel('Tinggi (cm):')
        self.height_label.setFont(header_font)
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Masukkan tinggi badan")
        self.height_input.setValidator(QDoubleValidator(0, 300, 2))
        self.layout.addWidget(self.height_label)
        self.layout.addWidget(self.height_input)

        self.weight_label = QLabel('Berat (kg):')
        self.weight_label.setFont(header_font)
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("Masukkan berat badan")
        self.weight_input.setValidator(QDoubleValidator(0, 300, 2))
        self.layout.addWidget(self.weight_label)
        self.layout.addWidget(self.weight_input)

        self.ap_hi_label = QLabel('Tekanan Darah Atas (ap_hi):')
        self.ap_hi_label.setFont(header_font)
        self.ap_hi_input = QLineEdit()
        self.ap_hi_input.setValidator(QIntValidator(0, 300))
        self.layout.addWidget(self.ap_hi_label)
        self.layout.addWidget(self.ap_hi_input)

        self.ap_lo_label = QLabel('Tekanan Darah Bawah (ap_lo):')
        self.ap_lo_label.setFont(header_font)
        self.ap_lo_input = QLineEdit()
        self.ap_lo_input.setValidator(QIntValidator(0, 200))
        self.layout.addWidget(self.ap_lo_label)
        self.layout.addWidget(self.ap_lo_input)

        self.cholesterol_label = QLabel('Kolesterol:')
        self.cholesterol_label.setFont(header_font)
        self.cholesterol_input = QComboBox()
        self.cholesterol_input.addItems(['1 = Normal', '2 = Above Normal', '3 = Well Above Normal'])
        self.layout.addWidget(self.cholesterol_label)
        self.layout.addWidget(self.cholesterol_input)

        self.gluc_label = QLabel('Glukosa:')
        self.gluc_label.setFont(header_font)
        self.gluc_input = QComboBox()
        self.gluc_input.addItems(['1 = Normal', '2 = Above Normal', '3 = Well Above Normal'])
        self.layout.addWidget(self.gluc_label)
        self.layout.addWidget(self.gluc_input)

        self.smoke_label = QLabel('Merokok:')
        self.smoke_label.setFont(header_font)
        self.smoke_input = QComboBox()
        self.smoke_input.addItems(['0 = Tidak', '1 = Ya'])
        self.layout.addWidget(self.smoke_label)
        self.layout.addWidget(self.smoke_input)

        self.alco_label = QLabel('Alkohol:')
        self.alco_label.setFont(header_font)
        self.alco_input = QComboBox()
        self.alco_input.addItems(['0 = Tidak', '1 = Ya'])
        self.layout.addWidget(self.alco_label)
        self.layout.addWidget(self.alco_input)

        self.active_label = QLabel('Aktif Fisik:')
        self.active_label.setFont(header_font)
        self.active_input = QComboBox()
        self.active_input.addItems(['0 = Tidak', '1 = Ya'])
        self.layout.addWidget(self.active_label)
        self.layout.addWidget(self.active_input)

        self.predict_button = QPushButton('Prediksi Risiko')
        self.predict_button.setIcon(QIcon(resource_path('predict_icon.png')))
        self.predict_button.setStyleSheet("background-color: lightblue; font-weight: bold;")
        self.predict_button.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_button)

        self.model_performance_button = QPushButton('Hasil Model')
        self.model_performance_button.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        self.model_performance_button.clicked.connect(self.show_model_performance)
        self.layout.addWidget(self.model_performance_button)

        self.setLayout(self.layout)

    def predict(self):
        try:
            age = int(self.age_input.text())
            gender = int(self.gender_input.currentText())
            height = float(self.height_input.text())
            weight = float(self.weight_input.text())
            ap_hi = float(self.ap_hi_input.text())
            ap_lo = float(self.ap_lo_input.text())
            cholesterol = int(self.cholesterol_input.currentText()[0])
            gluc = int(self.gluc_input.currentText()[0])
            smoke = int(self.smoke_input.currentText()[0])
            alco = int(self.alco_input.currentText()[0])
            active = int(self.active_input.currentText()[0])

            BMI = weight / ((height / 100) ** 2)

            input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, BMI]])
            input_data_scaled = scaler.transform(input_data)

            nb_prediction = nb.predict(input_data_scaled)

            rf_prediction = rf.predict(input_data_scaled)

            result_text = f"Naive Bayes Prediksi: {'Beresiko Cardio' if nb_prediction[0] == 1 else 'Tidak Beresiko Cardio'}\n"
            result_text += f"Random Forest Prediksi: {'Beresiko Cardio' if rf_prediction[0] == 1 else 'Tidak Beresiko Cardio'}"

            QMessageBox.information(self, 'Hasil Prediksi', result_text)

        except ValueError as e:
            QMessageBox.critical(self, 'Error', f'Invalid Input: {e}')
    ##menampilkan matrix performa akurasi
    def show_model_performance(self):
        model_performance_text = f"Naive Bayes Accuracy: {nb_accuracy:.2f}\n"
        model_performance_text += f"Random Forest Accuracy: {rf_accuracy:.2f}\n\n"
        model_performance_text += "Naive Bayes Classification Report:\n" + nb_class_report + "\n"
        model_performance_text += "Random Forest Classification Report:\n" + rf_class_report

        QMessageBox.information(self, 'Model Performance', model_performance_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CardiovascularApp()
    window.show()
    sys.exit(app.exec_())
