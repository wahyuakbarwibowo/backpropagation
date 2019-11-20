import numpy as np
import random
import math
from sklearn.datasets import load_digits
from keras.utils import to_categorical


class NeuralNetwork:
    def __init__(self, nInput, nHidden, nOutput, weights=[]):
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        if weights != []:
            self.weights = weights
        return

    def initWeight(self):
        weights = []
        #input -> hidden 1
        weights.append(np.zeros([self.nInput+1, self.nHidden[0]]))
        #hidden 1 -> nHidden
        for i in range(1, len(self.nHidden)):
            nNeuron = np.zeros([self.nHidden[i-1]+1, self.nHidden[i]])
            weights.append(nNeuron)
        #nHidden -> output
        weights.append(np.zeros([self.nHidden[-1]+1, self.nOutput]))
        
        #random weight
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    #weights[i][j][k] = 0.5
                    weights[i][j][k] = self.randomWeight()
        self.weights = weights

        hidden = np.array([[0.1,0.3,0.4],[0.2,0.4,0.15],[0.1,0.1,0.25],[0.15,0.3,0.4],[0.2,0.2,0.5]]) #bu indah
        output = np.array([[0.3,0.2,0.2],[0.05,0.1,0.1],[0.1,0.3,0.3],[0.4,0.4,0.4]]) #bu indah (tiga output)
        #output = np.array([[0.3,0.2],[0.05,0.1],[0.1,0.3],[0.4,0.4]]) #bu indah
        #hidden = np.array([[0.35, 0.35], [0.15, 0.25], [0.20, 0.30]]) #web
        #output = np.array([[0.60, 0.60], [0.40, 0.50], [0.45, 0.55]]) #web
        #self.weights = [hidden, output]
        #print('weights :', weights)
        return weights

    # random -1 to 1
    def randomWeight(self):
        return random.random() * 2 - 1

    def buildMatrix(self, x):
        newX = np.ones(len(x)+1)
        newX[1:] = x
        return newX

    def activation_sigmoid(self, nets):
        result = np.zeros(len(nets))
        for i in range(len(nets)):
            result[i] = 1 / (1 + math.exp(-nets[i]))
        return result

    def feedForwardPass(self, _input):
        result = [_input]
        for i in range(len(self.weights)):
            net = self.buildMatrix(result[i]).dot(self.weights[i])
            activation = self.activation_sigmoid(net)
            result.append(activation)
        #print('feedForwardPass: \n',result)
        return result

    def error(self, output, target):
        error = 1/2 * (target - output) ** 2
        result = 0
        for i in range(len(error)):
            result = result + error[i]
        return result
    
    def backwardPropagation(self, outputs, target, learningRate):
        output = outputs[-1]
        sigmaOutput = (target - output) * output * (1 - output)
        
        sigmas = [np.array(sigmaOutput)] # error -> net output
        #print('sigmas',sigmas)

        tempWeight = []
        for i in range(len(self.weights)-1, -1, -1):
            #print('wi', self.weights[i])

            delta = np.zeros(self.weights[i].shape)
            #print('delta', delta)
            output = self.buildMatrix(outputs[i]) # output layer sebelumnya + bias
            #print('==',output)
            for j in range(len(self.weights[i])):
                #print('wij',self.weights[i][j])
                for k in range(len(self.weights[i][j])):
                    #print('wijk',self.weights[i][j][k])
                    delta[j][k] = learningRate * sigmas[-1][k] * output[j]

            sigmas.append(sigmas[-1].dot(self.weights[i][1:].T) * outputs[i] * (1 - outputs[i]))
            #print('delta ----', delta)
            #print('sigma ----', sigmas)
            #print('w ----', self.weights[i] + delta)
            tempWeight = [self.weights[i] + delta] + tempWeight
            
        self.weights = tempWeight
        #print('backwardPropagation :\n',self.weights)
        return [tempWeight]

    def fit(self, data, target, epoch, learnRate):
        print('training...')
        self.initWeight()
        for i in range(epoch):
            print('-- epoch :', i+1, end=' ')
            for j in range(len(data)):
                feedForwardPass = self.feedForwardPass(data[j])
                backwardPropogation = self.backwardPropagation(feedForwardPass, target[j], learnRate)
            print('done')
        for i in range(len(self.weights)):
            self.weights[i] = list(self.weights[i])
        np.save('model.npy', self.weights)
        return self.weights

    def predict(self, data):
        print('testing...')
        result = []
        for i in range(len(data)):
            predict = self.feedForwardPass(data[i])[-1]
            maxInd = 0
            for j in range(len(predict)):
                if predict[j] > predict[maxInd]:
                    maxInd = j
            result.append(maxInd)
        return np.array(result)

if __name__ == "__main__":
    digits = load_digits()
    data = digits.data[:500]
    target = digits.target[:500]
    target_names = digits.target_names

    nInput = len(data[0])
    nOutput = len(target_names)
    nHidden = [nInput, nInput]

    epoch = 10
    learnRate = 0.05

    nn = NeuralNetwork(nInput, nHidden, nOutput)
    print('hidden layer : ', nHidden)
    print('epoch : ', epoch)
    print('learning Rate : ', learnRate)
    train = nn.fit(data, to_categorical(target), epoch, learnRate)
    predict = nn.predict(data)
    predictTrue = np.sum(predict == target)
    print('testing data true :', predictTrue, '/', len(predict))
    acc = predictTrue / len(predict) * 100
    print("testing accuracy :", acc, '%')

