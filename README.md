# GAN-hacks

Some useful GAN-hacks using Keras based on some GAN gyan from [Jason Brownlee](https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/)

1) Downsample using strided convolutions
```
from keras.models import Input, Model
from keras.layers import Conv2D

input_img = Input(shape = (64,64,1))
dis = Conv2D(64, (3,3), strides = (2,2), padding = 'same')(input_img)
model = Model(input_img, dis)
model.summary()
```

2) Upsample using strided convolutions
```
from keras.models import Input, Model
from keras.layers import Conv2DTranspose
input_img = Input(shape = (64, 64, 3))
us = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same')(input_img)
model = Model(input_img, us)
model.summary()
```

3) Use LeakyReLU
```
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.layers import LeakyReLU
input_img = Input(shape = (64, 64, 1))
dis = Conv2D(64, (3,3), strides = (2,2), padding = 'same')(input_img)
dis = LeakyReLU(0.2)(dis)
model = Model(input_img, dis)
model.summary()
```

4) Use Batch Normalization
```
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU
input_img = Input(shape = (64, 64, 1))
dis = Conv2D(64, (3,3), strides = (2,2), padding = 'same')(input_img)
dis = BatchNormalization()(dis)
dis = LeakyReLU(0.2)(dis)
model = Model(input_img, dis)
model.summary()
```

5) Use Gaussian Weight Initialization
```
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from keras.initializers import RandomNormal

input_img = Input(shape = (64, 64, 1))
init = RandomNormal(mean = 0.0, stddev = 0.02)
disc = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same', kernel_initializer = init)(input_img)
model = Model(input_img, disc)
```

6) Use Adam Stochastic Gradient Descent
```
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU
from keras.optimizers import Adam

input = Input(shape = (64, 64, 1))
disc = Conv2D(64, (3,3), strides = (2,2), padding = 'same')(input)
disc = BatchNormalization()(disc)
disc = LeakyReLU(0.2)(disc)
opt = Adam(lr = 0.0002, beta_1 = 0.5)
model = Model(input, disc)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
```

7) Scale Images in Range[-1, 1]
```
def scale_images(images):
	images = images.astype('float32')
	images = (images - 127.5)/127.5
	return images
```


8) Using Gaussian Latent Space
```
import numpy as np
def generate_latent_points(n_samples, latent_dim):
	x_input = np.random.randn(n_samples * latent_dim)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
x = generate_latent_points(100, 100)
```

9) Train the discriminator on separate batches of generator and discriminator
```
X_real, y_real = ....
discriminator.train_on_batch(X_real, y_real)
X_fake, y_fake = ...
discriminator.train_on_batch(X_fake, y_fake)
```

10) Make use of label smoothing rather than using actual label values
```
import numpy as np

def smooth_positive_labels(y):
	return y - 0.3 + (random(y.shape) * 0.5)

def smooth_negative_labels(y):
	return y + random(y.shape) * 0.3

y = np.ones((100, 1))
y = smooth_positive_labels(y)

z = np.zeros((100, 1))
z = smooth_negative_labels(z)
```




































