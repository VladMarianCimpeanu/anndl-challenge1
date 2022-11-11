# anndl
# TODO:
- DO NOT USE ImageGenerator() to create an iterator for the dataset, it is a bottleneck for the GPU.
- INSTEAD use Dataset object, there is a function prepare_batches() in [utils.py](https://github.com/VladMarianCimpeanu/anndl/blob/main/multiclassification_task/utils/utils.py) that creates a dataset object for the training and validation.
- for data augmentation use [preprocessing layers](https://keras.io/api/layers/preprocessing_layers/). These layers must belong to the model and put before the first convolutional layer.
- Consider large batches as possible when using GPU, it speeds up the training.
- species1 and species5 are the most difficult classes to learn, if even with data augmentation we can not reach good results, consider using weighted loss functions.
- training may be accelareted with other functions applied to the dataset object, info [here](https://www.tensorflow.org/guide/data_performance). It shouldn't be necessary, but I may have a look.
