# S7-Assignment-Solution

This repository contains three different Convolutional Neural Network (CNN) models implemented for the MNIST dataset. The goal was to achieve a test accuracy of 99.4% or higher with less than 8000 parameters within a maximum of 15 epochs.

### Model 1

**Target**:

1. Get the set-up right
2. Set Transforms
3. Set Data Loader
4. Set Basic Working Code
5. Set Basic Training  & Test Loop

**Results**:

1. Parameters: 416K
2. Best Training Accuracy: 99.73
3. Best Test Accuracy: 99.08

**Analysis**:

1. Model for such a problem but is working
2. Model is over-fitting



### Model 2

**Target**:

1. Make the model lighter
2. Add Batch-norm, Regularization, Dropout to increase model efficiency and decrease the difference between train and test accuracy
3. Adding maxpool after receptive field of 7 and second maxpool after receptive field of 20
4. Add GAP and remove the last BIG kernel.

**Results**:

1. Parameters: 7.7k
2. Best Training Accuracy: 99.9
3. Best Test Accuracy: 99.25

**Analysis**:

1. Good model!
2. Regularization working. 
3. Model is capable if pushed further, possibly by adding another maxpool as seen from incorrect classified images, some struggle with slightly different shape of the digit. Also by adding image augmentation.



### Model 3

**Target**:

1. Increase model capacity. Add layers and another maxpool. So we have 3 blocks each with 3 conv and 1 maxpool.
2. Reduce the output size to limit the parameters to less than 8k
3. Add image augmentations rotation with +-7 degrees, centercrop with size 22 and probability of 10%, and colorjitter with default contrast values.
4. Increase LR to 0.1 and add ReduceLROnPlateau with patience/stepsize = 3 epochs.

**Results**:

1. Parameters: 7.9k
2. Best Training Accuracy: 100.0
3. Best Test Accuracy: 99.45

**Analysis**:

1. Adding layers and maxpool helped model form parts of objects better and combine them to classify object which is our digit.
2. The test accuracy is up so image augmentation helped
3. LR scheduler helped in reaching 99.4 in 10 epochs
4. Possible overfitting as train accuracy reached 100%, maybe increase dropout from 5 to 10%. Playing around more with LR scheduler might help to get > 99.5%


Overall, these three models showcase the progress made in achieving the target accuracy of 99.4% on the MNIST dataset while maintaining parameter limits. The models demonstrate the importance of regularization techniques, model capacity, and the impact of image augmentation in enhancing performance. Further fine-tuning and experimentation can lead to even higher accuracy.
