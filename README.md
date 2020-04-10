CNN - VGG16 diagnostic classifier
==================================

Covid19 image detector

Dataset provided by:https://www.pyimagesearch.com/

VGG16 code architecture: https://www.pyimagesearch.com/

Diagnosis system based on neural networks, for detection of covid19 based
on radiographs.

Thanks to pyimagesearch for teaching keras+tensorflow, and image processing
using open computer vision (cv)

Disclaimer
----------
This work is for didactic purposes and should not be used for real diagnosis


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

    $ python3 train_covid19.py --dataset dataset --model model/covid19.model --labelbin labels/labels --plot plot/plot-covid19.png

Test & classify
---------------
See dataset/covid(+) & normal (-)

::

    $ python3 classify_covid19.py --model model/covid19.model  --labelbin labels/labels --image test/positive/ryct.2020200034.fig5-day7.jpeg

::

    $ python3 classify_covid19.py --model model/covid19.model  --labelbin labels/labels --image test/negative/person1935_bacteria_4849.jpeg


License
-------
MIT license

Nacho Ariza apr.2020


