import numpy as np

import theano
import theano.tensor as T


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def set_all_param_values(layers, values, order):
    params = []
    for layerName in order:
        layer = layers[layerName]
        if layer.params is not None:
            params.extend(layer.params)
    if len(params) != len(values):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))
    for p, v in zip(params, values):
        if p.get_value().shape != v.shape:
            raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                             (p.get_value().shape, v.shape))
        else:
            p.set_value(v)

# def change_input(nets, value, order):
#     for x in order:
#         if x == 'input':
#             layer = nets[x]
#             layer.output = value
#         else:
#             layer = nets[x]
#             layer.update_output()

def change_input2(inputlayer, value):
    inputlayer.output = value
    inputlayer.propagate_output()


def get_outputs(layers, inputs):
    # inputs: dict{ inputlayer: input}
    for inputlayer in inputs:
        change_input2(inputlayer, inputs[inputlayer])
    return [v.output for v in layers.values()]
