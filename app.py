from flask import Flask, render_template, request
import numpy as np
from sklearn.datasets import load_digits
from MLP import NeuralNetwork
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image
import cv2

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    if request.method == 'POST':
        mode = request.form['mode']
        if mode == 'Training':
            test_size = (100 - int(request.form['komposisi'])) / 100
            epochs = int(request.form['epochs'])
            learning_rate = float(request.form['learningrate'])

            digits = load_digits()
            target_names = digits.target_names
            simpan_data_test = digits.data[0:10, :]
            np.save('data_test.npy', simpan_data_test)
            data_training, data_test, target_training, target_test = \
                train_test_split(digits.data, digits.target, test_size=test_size, random_state=0)
            nInput = len(data_training[0])
            nOutput = len(target_names)
            nHidden = [nInput, nInput]
            nn = NeuralNetwork(nInput, nHidden, nOutput)
            train = nn.fit(data_training, to_categorical(target_training), epochs, learning_rate)

            predict = nn.predict(data_test)
            predictTrue = np.sum(predict == target_test)
            akurasi = predictTrue / len(predict) * 100
            

            # np.save('model.npy', train)
            return render_template('index.html', mode='Testing', hasil_training=akurasi)
        else:
            # path_hasil = 'static/assets/hasil/'
            if request.form['optradioimg'] != '' and request.files['query_img_upload'].filename != '':
                file = request.files['query_img_upload']

                img = Image.open(file.stream)
                uploaded_img_path = 'static/assets/upload/'+ file.filename
                result_name = file.filename[:-4]

                img.save(uploaded_img_path)

                img_input = cv2.imread(uploaded_img_path)
                gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
                gray_double = np.array(gray, dtype=float)
                data = np.reshape(gray_double, (gray_double.shape[0]*gray_double.shape[1]), 1)

                nInput = 64
                nOutput = 10
                nHidden = [nInput, nInput]
                train = np.load('model.npy', allow_pickle=True)
                nn = NeuralNetwork(nInput, nHidden, nOutput, train)

                
                predict = nn.predict(data)
                return render_template('index.html', url_image=uploaded_img_path, kelas=predict)

            elif request.form['optradioimg'] != '':
                data_test = np.load('data_test.npy')
                gambar_ke = int(request.form['optradioimg'])
                uploaded_img_path = 'static/assets/img/' + str(gambar_ke) + '.png'
                train = np.load('model.npy', allow_pickle=True)

                nInput = 64
                nOutput = 10
                nHidden = [nInput, nInput]
                nn = NeuralNetwork(nInput, nHidden, nOutput, train)
                predict = nn.predict(data_test[gambar_ke: gambar_ke + 1, :])
                return render_template('index.html', url_image=uploaded_img_path, kelas=predict)

            else:
                file = request.files['query_img_upload']

                img = Image.open(file.stream)
                uploaded_img_path = 'static/assets/upload/'+ file.filename
                result_name = file.filename[:-4]

                img.save(uploaded_img_path)

                img_input = cv2.imread(uploaded_img_path)
                gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
                gray_double = np.array(gray, dtype=float)
                data = np.reshape(gray_double, (gray_double.shape[0]*gray_double.shape[1]), 1)

                nInput = 64
                nOutput = 10
                nHidden = [nInput, nInput]
                train = np.load('model.npy', allow_pickle=True)
                nn = NeuralNetwork(nInput, nHidden, nOutput, train)
                
                predict = nn.predict(data)
                return render_template('index.html', url_image=uploaded_img_path, kelas=predict)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()