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

2) 