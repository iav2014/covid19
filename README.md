CNN - VGG16 diagnostic classifier
==================================

Covid19 image detector

Dataset provided by:https://www.pyimagesearch.com/

VGG16 code architecture: https://www.pyimagesearch.com/

Diagnosis system based on neural networks, for detection of covid19 based
on radiographs.

COVID19Net: 

shells:

	pooling2D

	Flatten

	Dense -> relu

	Dropout

	Dense -> softmax


Image are taken from

https://github.com/ieee8023/covid-chestxray-dataset

You can test & train with other covid19 dataset, like

https://github.com/UCSD-AI4H/COVID-CT

https://www.kaggle.com/tawsifurrahman/covid19-radiography-database/data


Thanks to pyimagesearch for teaching keras+tensorflow, and image processing
using open computer vision (cv)

Disclaimer
----------
This work is for didactic purposes (train & classify neural network) and should not be used for real diagnosis


Requirements
--------------
-  Install `tensorflow + keras`_

https://www.tensorflow.org/install
https://keras.io/

-  opencv
https://opencv.org/



Train: generate model
---------------------
Estimated time: 160s with macbookpro 2017, 15" 16Gb i7 ssd

With GPU: 107s.

::

    $ python3 train_covid19.py --dataset dataset --model model/covid19.model --labelbin labels/covid19.labels --plot plot/plot-covid19.png

Test & classify
---------------
See dataset/covid(+) & normal (-)

::

    $ python3 classify_covid19.py --model model/covid19.model  --labelbin labels/covid19.labels --image test/covid19/1-s2.0-S0140673620303706-fx1_lrg.jpg

::

    $ python3 classify_covid19.py --model model/covid19.model  --labelbin labels/labels --image test/normal/IM-0033-0001-0001.jpeg



Training using Macbook pro GPU Radeon pro 560
---------------------------------------------

Read this first: https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545

::

    $ pip install pyopencl

::

    $ pip install plaidml-keras

::

    $ plaidml-setup

::

    $ python3 trainGPU.py --dataset dataset --model model/covid19GPU.model --labelbin labels/covid19.labels --plot plot/plot-covid19.png


Test & classify
---------------
See dataset/covid(+) & normal (-)

::

    $ python3 classify_gpu_covid19.py --model model/covid19GPU.model  --labelbin labels/covid19.labels --image test/covid19/1-s2.0-S0140673620303706-fx1_lrg.jpg

::

    $ python3 classify_gpu_covid19.py --model model/covid19GPU.model  --labelbin labels/labels --image test/normal/IM-0033-0001-0001.jpeg



Install 
License
-------
MIT license

Nacho Ariza apr.2020


