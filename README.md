CNN - VGG16 diagnostic classifier
==================================

Covid19 image detector

Dataset provided by:https://www.pyimagesearch.com/

VGG16 code architecture: https://www.pyimagesearch.com/

Diagnosis system based on neural networks, for detection of covid19 based
on radiographs.

Image are taken from

https://github.com/ieee8023/covid-chestxray-dataset

You can test & train with other covid19 dataset, like

https://github.com/UCSD-AI4H/COVID-CT


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
Estimated time: 7 min
macbookpro 2017, 15" 16Gb i7 ssd

::

    $ python3 train_covid19.py --dataset dataset --model model/covid19.model --labelbin labels/covid19.labels --plot plot/plot-covid19.png

Test & classify
---------------
See dataset/covid(+) & normal (-)

::

    $ python3 classify_covid19.py --model model/covid19.model  --labelbin labels/covid19.labels --image test/covid19/1-s2.0-S0140673620303706-fx1_lrg.jpg

::

    $ python3 classify_covid19.py --model model/covid19.model  --labelbin labels/labels --image test/normal/IM-0033-0001-0001.jpeg


License
-------
MIT license

Nacho Ariza apr.2020


