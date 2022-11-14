# anndl
# NEW: by performing some experiments I noticed that going deeper with fully connected layers does not improve the performances, instead, by changing the convolution part we get better results: it seems that the problem is given by the extraction of features which seems to be too weak. Since images are really small, we can use very small amount of maxpooling layers before we get an output with just one dimension (channel depth), so in order to increase the complexity of the convolution architecture, we can **not** put max pooling layers after each convolution layer. By stacking different convolution layers (with non linear activation functions) it seems to work pretty well.
# TODO:
- DO NOT USE ImageGenerator() to create an iterator for the dataset, it is a bottleneck for the GPU.
- INSTEAD use Dataset object, there is a function prepare_batches() in [utils.py](https://github.com/VladMarianCimpeanu/anndl/blob/main/multiclassification_task/utils/utils.py) that creates a dataset object for the training and validation.
- for data augmentation use [preprocessing layers](https://keras.io/api/layers/preprocessing_layers/). These layers must belong to the model and put before the first convolutional layer.
- Consider large batches as possible when using GPU, it speeds up the training.
- species1 and species5 are the most difficult classes to learn, if even with data augmentation we can not reach good results, consider using weighted loss functions.
- training may be accelareted with other functions applied to the dataset object, info [here](https://www.tensorflow.org/guide/data_performance). It shouldn't be necessary, but I may have a look.
