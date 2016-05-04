This autoencoder.py implements the deep autoencoder network. Interfaces of the autoencoder is the same as sklearn's Manifold Learning.

* fit(X) Fit the autoecoder network for data X
* fit_transform(X)   Fit the model from data in X and transform X.
* get_params()  Get parameters of this network.
* reconstruction_error(X)  Compute the reconstruction error for the data X.
* set_params()    Set the parameters which comes from saved file.
* transform(X)    Transform X.

The test.py is an example for reducing the 28*28 mnist dataset images into 2 dimention and visualize it.

I change some code from [Variational Autoencoder in TensorFlow](https://jmetzen.github.io/2015-11-27/vae.html) to this autoencoder network in tensorflow.
