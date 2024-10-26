import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Definirea funcției de activare sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivata funcției de activare sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, stratul_de_intrare, stratul_ascuns, stratul_de_iesire):
        self.stratul_de_intrare = stratul_de_intrare
        self.stratul_ascuns = stratul_ascuns
        self.stratul_de_iesire = stratul_de_iesire

        self.weights_input_hidden = np.random.randn(self.stratul_de_intrare, self.stratul_ascuns)
        self.weights_hidden_output = np.random.randn(self.stratul_ascuns, self.stratul_de_iesire)

    def propagare_inainte(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output = sigmoid(self.output_layer_input)
        return self.output

    def propagare_inapoi(self, X, y, output, learning_rate):
        self.error = y - output

        delta_output = self.error * sigmoid_derivative(output)
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, delta_output) * learning_rate

        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_layer_output)
        self.weights_input_hidden += np.dot(X.T, delta_hidden) * learning_rate

    def antrenare(self, X, y, epoci, learning_rate):
        erori = []
        for epoca in range(epoci):
            eroare_de_epoca = 0
            for i in range(len(X)):
                output = self.propagare_inainte(X[i].reshape(1, -1))
                self.propagare_inapoi(X[i].reshape(1, -1), y[i].reshape(1, -1), output, learning_rate)
                pierdere = np.mean(np.square(y[i].reshape(1, -1) - output))
                eroare_de_epoca += pierdere
            erori.append(eroare_de_epoca / len(X))
            plt.plot(range(epoca + 1), erori, color='blue')
            plt.xlabel('Epoci')
            plt.ylabel('Eroare')
            plt.title('Eroare vs. Epoci')
            plt.text(epoca + 0.5, erori[-1], f'Eroare = {erori[-1]:.6f}', fontsize=8, color='red', ha='right')
            plt.pause(0.001)
            if epoca > 0:
                plt.gca().texts[-2].remove()
        plt.show()

    def test(self, X_test, y_test):
        predictii_corecte = 0
        for i in range(len(X_test)):
            predictie = self.propagare_inainte(X_test[i].reshape(1, -1))
            clasa_prezisa = 1 if predictie >= 0.5 else 0
            if clasa_prezisa == y_test[i]:
                predictii_corecte += 1
        acuratete = predictii_corecte / len(X_test)
        return acuratete

def incarca_imaginile(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            images.append(image_path)
    return images


def pregatirea_imaginilor(image_path):
    image = Image.open(image_path).resize((50, 50)).convert("L")
    image_array = np.array(image)
    return image_array.flatten() / 255.0


def incarca_datele():
    X = []
    y = []

    no_escooter_images = incarca_imaginile("no_escooter")
    for image_path in no_escooter_images:
        X.append(pregatirea_imaginilor(image_path))
        y.append(0)

    with_escooter_images = incarca_imaginile("with_escooter")
    for image_path in with_escooter_images:
        X.append(pregatirea_imaginilor(image_path))
        y.append(1)

    return np.array(X), np.array(y)


# Divizarea datelor în seturi de antrenament și de testare
X, y = incarca_datele()
X_antrenare, X_test, y_antrenare, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

stratul_de_intrare = 2500
stratul_ascuns = 250
stratul_de_iesire = 1
nn = NeuralNetwork(stratul_de_intrare, stratul_ascuns, stratul_de_iesire)
nn.antrenare(X_antrenare, y_antrenare.reshape(-1, 1), epoci=100, learning_rate=0.1)

acuratete = nn.test(X_test, y_test)
print(f'Acuratețea pe setul de testare: {acuratete * 100:.2f}%')