import numpy as np
import math

w11 = 0.2
w12 = 0.2

w13 = -0.2
w14 = -0.2

w15 = 0.2
w16 = 0.2

w17 = 0.3
w18 = 0.3

w21 = -0.5
w22 = 0
w23 = 0.25
w24 = 0.5

bias = 2
lamb = 0.1

standard = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]

"""Создается нейронка вида
        *

    *   *
            *
    *   *

        *
"""


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def layer_result(inputs, weights):
    return np.dot(np.matrix(inputs), np.matrix(weights))


def sigma_yong_layer(gradient, weight, neuron_output):
    return gradient * weight * neuron_output * (1 - neuron_output)


sigmoid_vec = np.vectorize(sigmoid)

# def gradient(error, out):
# return error * (math.exp(-out) / (1 + math.exp(-out)) ^ 2)

for epoch in range(100000):
    for stage in standard:
        hidden_layer_input = layer_result([stage[0], stage[1]], [[w11, w13, w15, w17], [w12, w14, w16, w18]])
        hidden_layer_output = sigmoid_vec(hidden_layer_input)
        out_layer_input = layer_result(hidden_layer_output, [[w21], [w22], [w23], [w24]])
        out_layer_result = sigmoid_vec(out_layer_input)

        error = stage[2] - out_layer_result.item()
        error1 = error * w21
        error2 = error * w22
        error3 = error * w23
        error4 = error * w24

        hd_out = hidden_layer_output.getA()[0]
        w11 = w11 + error1 * hd_out[0] * (1 - hd_out[0]) * stage[0] * lamb
        w12 = w12 + error1 * hd_out[0] * (1 - hd_out[0]) * stage[1] * lamb

        w13 = w13 + error2 * hd_out[1] * (1 - hd_out[1]) * stage[0] * lamb
        w14 = w14 + error2 * hd_out[1] * (1 - hd_out[1]) * stage[1] * lamb

        w15 = w15 + error3 * hd_out[2] * (1 - hd_out[2]) * stage[0] * lamb
        w16 = w16 + error3 * hd_out[2] * (1 - hd_out[2]) * stage[1] * lamb

        w17 = w17 + error4 * hd_out[3] * (1 - hd_out[3]) * stage[0] * lamb
        w18 = w18 + error4 * hd_out[3] * (1 - hd_out[3]) * stage[1] * lamb

        neural_out = out_layer_result.item()
        w21 = w21 + error * neural_out * (1 - neural_out) * hd_out[0] * lamb
        w22 = w22 + error * neural_out * (1 - neural_out) * hd_out[1] * lamb
        w23 = w23 + error * neural_out * (1 - neural_out) * hd_out[2] * lamb
        w24 = w24 + error * neural_out * (1 - neural_out) * hd_out[3] * lamb

        print('-----')
        print(w11)
        print(w21)
        print('-----')


def neural_network(i1, i2):
    hidden_layer_input = layer_result([i1, i2], [[w11, w13, w15, w17], [w12, w14, w16, w18]])
    hidden_layer_output = sigmoid_vec(hidden_layer_input)
    out_layer_input = layer_result(hidden_layer_output, [[w21], [w22], [w23], [w24]])
    out_layer_result = sigmoid_vec(out_layer_input)
    return round(out_layer_result.item())


print(neural_network(0, 0))
print(neural_network(0, 1))
print(neural_network(1, 0))
print(neural_network(1, 1))
