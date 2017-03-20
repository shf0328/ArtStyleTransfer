import theano
import theano.tensor as T
import numpy
import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import pickle

from ImageUtils import ImageHelper
from Layers import InputLayer, ConvLayer, PoolLayer
from utils import floatX, set_all_param_values, get_outputs

rng = numpy.random.RandomState(23455)

IMAGE_W = 600


###############################################################
# build the model
###############################################################
def runSGD(cl_weight, con_layers, sty_layers, photopath, artpath):
    def build_and_load_model():
        def build_model(theano_input):
            net = {}
            order = ['input', 'conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1',
                     'conv3_2', 'conv3_3', 'conv3_4', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                     'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
            net['input'] = InputLayer(theano_input, (1, 3, IMAGE_W, IMAGE_W))
            net['conv1_1'] = ConvLayer(net['input'], 64, 3, rng, flip_filters=False)
            net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, rng, flip_filters=False)
            net['pool1'] = PoolLayer(net['conv1_2'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, rng, flip_filters=False)
            net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, rng, flip_filters=False)
            net['pool2'] = PoolLayer(net['conv2_2'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, rng, flip_filters=False)
            net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, rng, flip_filters=False)
            net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, rng, flip_filters=False)
            net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, rng, flip_filters=False)
            net['pool3'] = PoolLayer(net['conv3_4'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, rng, flip_filters=False)
            net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, rng, flip_filters=False)
            net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, rng, flip_filters=False)
            net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, rng, flip_filters=False)
            net['pool4'] = PoolLayer(net['conv4_4'], poolsize=(2, 2), mode='average_exc_pad')
            net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, rng, flip_filters=False)
            net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, rng, flip_filters=False)
            net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, rng, flip_filters=False)
            net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, rng, flip_filters=False)
            net['pool5'] = PoolLayer(net['conv5_4'], poolsize=(2, 2), mode='average_exc_pad')
            return net, order

        # build it
        net, order = build_model(T.tensor4())
        # load it
        values = pickle.load(open('./data/vgg19_normalized.pkl', 'rb'))['param values']
        set_all_param_values(net, values, order)
        return net

    net = build_and_load_model()

    layers = con_layers + sty_layers
    layers = {k: net[k] for k in layers}
    # select the layer to use
    # layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    # layers = {k: net[k] for k in layers}

    ###############################################################
    # get the images
    ###############################################################
    imageHelper = ImageHelper(IMAGE_W=600)
    photo, art = imageHelper.prep_photo_and_art(photo_path=photopath, art_path=artpath)

    input_im_theano = T.tensor4()
    # compute layer activations for photo and artwork
    outputs = get_outputs(layers, {net['input']: input_im_theano})
    # these features are constant which is the reference for loss
    photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                      for k, output in zip(layers.keys(), outputs)}
    art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                    for k, output in zip(layers.keys(), outputs)}

    ###############################################################
    # calculate loss and grads
    ###############################################################
    # Get expressions for layer activations for generated image
    generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))
    gen_features = get_outputs(layers, {net['input']: generated_image})
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

    def gram_matrix(x):
        x = x.flatten(ndim=3)
        g = T.tensordot(x, x, axes=([2], [2]))
        return g

    def content_loss(P, X, layer):
        p = P[layer]
        x = X[layer]

        loss = 1. / 2 * ((x - p) ** 2).sum()
        return loss

    def style_loss(A, X, layer):
        a = A[layer]
        x = X[layer]

        A = gram_matrix(a)
        G = gram_matrix(x)

        N = a.shape[1]
        M = a.shape[2] * a.shape[3]

        loss = 1. / (4 * N ** 2 * M ** 2) * ((G - A) ** 2).sum()
        return loss

    def total_variation_loss(x):
        return (
        ((x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 + (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2) ** 1.25).sum()

    # Define loss function
    losses = []

    # content loss
    losses.append(cl_weight * content_loss(photo_features, gen_features, 'conv4_2'))
	
	# style loss
    for style_layer in sty_layers:
        losses.append(0.2e6 * style_loss(art_features, gen_features, style_layer))

    
    

    # total variation penalty
    losses.append(0.1e-7 * total_variation_loss(generated_image))

    total_loss = sum(losses)

    grad = T.grad(total_loss, generated_image)

    ###############################################################
    # start to optimize
    ###############################################################

    def RMSprop(cost, params, lr=0.8, rho=0.95, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates

    updates = RMSprop(cost=total_loss, params=[generated_image])

    train_model = theano.function([], total_loss, updates=updates)

    for i in range(3000):
        print train_model()

    ###############################################################
    # display result
    ###############################################################

    xout = generated_image.get_value().astype('float64')
    plt.figure(figsize=(4, 4))
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.imshow(imageHelper.deprocess(xout))
    plt.show()
    return imageHelper.deprocess(xout)