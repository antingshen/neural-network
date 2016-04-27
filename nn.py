import numpy as np
import random
import itertools
import pickle
import math
import sys

from tqdm import tqdm, trange
from pdb import set_trace as trace
import scipy.io as sio
from sklearn.utils import shuffle


def gaussian_initializer():
    sigma = 0.01
    return random.gauss(0, sigma)


class FullyConnectedLayer:

    def __init__(self, *, num_input, num_output, initializer=gaussian_initializer):
        self.num_input = num_input
        self.num_output = num_output
        self.weights = np.empty((num_output, num_input+1))
        for x in np.nditer(self.weights, op_flags=['readwrite']):
            x[...] = initializer()

    def forward_pass(self, inputs):
        assert inputs.shape[0] == self.num_input
        inputs_with_bias = np.insert(inputs, len(inputs), 1, axis=0)
        self.inputs = inputs_with_bias
        outputs = np.dot(self.weights, inputs_with_bias) 
        assert outputs.shape[0] == self.num_output
        return outputs

    def backward_pass(self, prev_gradient):
        next_gradient = np.dot(self.weights.T, prev_gradient)
        update = np.dot(prev_gradient, self.inputs.T)
        self.weights = self.weights * (1 - self.learning_rate * self.decay) - self.learning_rate * update
        next_gradient = next_gradient[:-1]
        assert len(next_gradient) == self.num_input
        return next_gradient


class BatchNormLayer:

    alpha = 0.0000001

    def __init__(self, num_input):
        self.total_means = np.zeros((num_input,))
        self.total_stddevs = np.zeros((num_input,))
        self.total = 0

    def forward_pass(self, inputs):
        self.inputs = inputs
        if len(inputs.shape) > 1:
            self.means = np.mean(inputs, axis=1)
            self.stddevs = np.std(inputs, axis=1)
            self.total_means += self.means
            self.total_stddevs += self.stddevs
            self.total += 1
            self.means = self.means[None].T
            self.stddevs = self.stddevs[None].T
        else:
            self.means = self.total_means / self.total
            self.stddevs = self.total_stddevs / self.total
        ret = (inputs - self.means) / (self.stddevs + self.alpha) # TODO: optimize
        assert ret.shape == inputs.shape
        trace()
        return ret

    def backward_pass(self, prev_gradient): # NOT ACCURATE, TODO
        # stddev_grad = (self.inputs - self.means)
        ret = prev_gradient / self.stddevs
        assert ret.shape == prev_gradient.shape
        return ret


class BatchNormScaleLayer:

    def __init__(self):
        self.scale = 1
        self.bias = 0

    def forward_pass(self, inputs):
        self.inputs = inputs
        return inputs * self.scale + self.bias

    def backward_pass(self, prev_gradient):
        next_gradient = prev_gradient * self.scale
        self.bias = self.bias * (1 - self.learning_rate * self.decay) - self.learning_rate * np.sum(prev_gradient)
        self.scale = self.scale * (1 - self.learning_rate * self.decay) - self.learning_rate * np.sum(np.multiply(self.inputs, prev_gradient))
        return next_gradient


class SigmoidLayer:

    def forward_pass(self, inputs):
        inputs = np.exp(-inputs)
        outputs = 1/(1+inputs)
        self.outputs = outputs
        return outputs

    def backward_pass(self, prev_gradient):
        gradient = np.multiply(self.outputs, 1-self.outputs)
        return np.multiply(prev_gradient, gradient)


class TanhLayer:
    
    def forward_pass(self, inputs):
        outputs = np.tanh(inputs)
        self.outputs = outputs
        return outputs

    def backward_pass(self, prev_gradient):
        gradient = 1 - np.multiply(self.outputs, self.outputs)
        return np.multiply(prev_gradient, gradient)


class LossLayer:

    def __init__(self, *, num_input):
        self.num_input = num_input
        self.correct = 0
        self.total = 0

    def get_accuracy(self):
        acc = self.correct / self.total
        self.correct = 0
        self.total = 0
        return acc

    def forward_pass(self, samples, labels):
        predictions = np.argmax(samples, axis=0)
        self.total += len(predictions)
        self.correct += np.sum(predictions == labels)
        return self.forward_pass_loss(samples, labels)


# class MeanSquaredErrorLoss(LossLayer):

#     def forward_pass_loss(self, features, label):
#         correct = np.zeros(self.num_input)
#         correct[label] = 1
#         self.half_gradient = features - correct
#         error = sum((features - correct)**2) / 2
#         return error

#     def backward_pass(self):
#         return self.half_gradient * 2


class CrossEntropyErrorLoss(LossLayer):

    alpha = 0.000000001

    def forward_pass_loss(self, samples, labels):
        assert samples.shape[0] == self.num_input
        total = 0
        self.neg_gradient = np.empty((samples.shape))
        for feat_idx, feature in enumerate(samples):
            for samp_idx, value in enumerate(feature):
                if feat_idx == labels[samp_idx]:
                    total += math.log(value + self.alpha)
                    self.neg_gradient[feat_idx][samp_idx] = 1 / (value + self.alpha)
                else:
                    total += math.log(1 - value + self.alpha)
                    self.neg_gradient[feat_idx][samp_idx] = 1 / (value - 1 + self.alpha)
        return -total

    def backward_pass(self):
        return -self.neg_gradient

class NeuralNetwork:

    def __init__(self, layers, loss, learning_rate, 
            maxsamples=40000*100, batch_size=1, print_freq=5000, val_freq=40000, log='train.log'):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.maxiters = maxsamples // batch_size
        self.batch_size = batch_size
        self.print_freq = print_freq // batch_size
        self.val_freq = val_freq // batch_size
        self.log = log
        self.decay = self.learning_rate.pop('decay')
        for layer in self.layers:
            layer.decay = self.decay

    def set_data(self, train_data, train_labels, val_data=[], val_labels=[], test_data=[]):
        self.calculate_preprocess(train_data)
        self.train_data = self.apply_preprocess(train_data)
        self.train_labels = train_labels
        self.val_data = self.apply_preprocess(val_data)
        self.val_labels = val_labels
        self.test_data = self.apply_preprocess(test_data)

    def classify(self, features):
        return np.argmax(self.forward_pass(features))

    def forward_pass(self, features):
        for layer in self.layers:
            features = layer.forward_pass(features)
        return features

    def gradient_check(self, features, label):
        raise NotImplementedError

    def calculate_preprocess(self, data):
        self.means = np.mean(data, axis=0)
        self.stddevs = np.std(data, axis=0)

    def apply_preprocess(self, data):
        data = np.subtract(data, self.means)
        data = data / (self.stddevs + 0.00000001)
        return data

    def train_samples(self, start, end):
        features = self.train_data[start:end].T
        labels = self.train_labels[start:end]
        assert features.shape[1]
        features = self.forward_pass(features)
        loss = self.loss.forward_pass(features, labels)
        gradient = self.loss.backward_pass()
        for layer in reversed(self.layers):
            gradient = layer.backward_pass(gradient)
        return loss / (end - start)

    def train(self):
        logfile = open(self.log, 'w')
        try:
            num_train = len(self.train_labels)
            iters = 0
            loss = 0
            i = num_train
            for iters in trange(self.maxiters):
                if iters in self.learning_rate:
                    for layer in self.layers:
                        layer.learning_rate = self.learning_rate[iters]
                if i >= num_train:
                    i = 0
                    self.train_data, self.train_labels = shuffle(self.train_data, self.train_labels)
                loss += self.train_samples(i, i + self.batch_size)
                i += self.batch_size
                if iters % self.print_freq == 0:
                    loss /= self.print_freq
                    validation_accuracy = '\b'*8 + ' '*14
                    training_accuracy = self.loss.get_accuracy()

                    if len(self.test_data) > 0 and (iters % num_train == 0 or training_accuracy >= 0.999):
                        self.output_test('{}.csv'.format(iters))
                    if iters % self.val_freq == 0 and iters > 0:
                        if len(self.val_data) > 0:
                            validation_accuracy = '{:.4f}'.format(self.validate())
                    
                    status = '{: >9} - loss: {:.4f} - train: {:.4f} - val: {}{: >26}'.format(
                        iters, loss, training_accuracy, validation_accuracy, ' ')
                    print('\r' + status)
                    print(status, file=logfile)
                    logfile.flush()
                    loss = 0
        except KeyboardInterrupt:
            pass
        logfile.close()

    def validate(self):
        correct = 0
        for features, label in zip(self.val_data, self.val_labels):
            if self.classify(features) == label:
                correct += 1
        return correct / len(self.val_labels)

    def output_test(self, name='submission.csv'):
        with open('submissions/' + name, 'w') as f:
            print('Id,Category', file=f)
            for i,sample in enumerate(self.test_data):
                label = self.classify(sample)
                print('{},{}'.format(i+1, label), file=f)


def make_net():
    layers = [
        FullyConnectedLayer(num_input=784, num_output=200),
        TanhLayer(),
        FullyConnectedLayer(num_input=200, num_output=10),
        SigmoidLayer(),
    ]
    batch_1_lr = {
        'decay': 0.0008,
              0: 0.005,
          80000: 0.004,
         240000: 0.003,
        2400000: 0.002,
        3600000: 0.001,
    }
    batch_200_lr = {
        'decay': 0.0003,
              0: 0.005,
           2000: 0.004,
          24000: 0.003,
         240000: 0.002,
         360000: 0.001,
    }
    return NeuralNetwork(layers, CrossEntropyErrorLoss(num_input=10), batch_200_lr, 
        batch_size=200, print_freq=20000, val_freq=120000)


def load_digits_data(num_train, num_val):
    total_samples = 60000
    train_mat = sio.loadmat('dataset/train.mat')

    def set_image(array, all_images, index):
        for i, j in itertools.product(range(28), repeat=2):
            array[i+28*j] = all_images[j][i][index]

    train = (np.empty([num_train, 28*28]), np.empty(num_train))
    val = (np.empty([num_val, 28*28]), np.empty(num_val))
    for samples_remaining in range(total_samples-1, -1, -1):
        if num_val == num_train == 0:
            break
        roll = random.random()
        if roll < num_val / (samples_remaining + 1):
            num_val -= 1
            set_image(val[0][num_val], train_mat['train_images'], samples_remaining)
            val[1][num_val] = train_mat['train_labels'][samples_remaining]
        elif roll < (num_val + num_train) / (samples_remaining + 1):
            num_train -= 1
            set_image(train[0][num_train], train_mat['train_images'], samples_remaining)
            train[1][num_train] = train_mat['train_labels'][samples_remaining]
    print('data loaded')
    return train, val

def load_digits_test():
    test_mat = sio.loadmat('dataset/test.mat')
    test_images = []
    for image in test_mat['test_images']:
        test_images.append(np.ndarray.flatten(image))
    return test_images


def mnist():
    train, val = pickle.load(open('trainval.pickle', 'rb'))
    # with open('trainval.pickle', 'wb') as f:
    #     train, val = load_digits_data(40000, 20000)
    #     pickle.dump((train, val), f)

    test_data = load_digits_test()
    net = make_net()
    net.set_data(train[0], train[1], val[0], val[1], test_data)

    net.train()
    net.output_test()


if __name__ == '__main__':
    random.seed(0)
    mnist()
