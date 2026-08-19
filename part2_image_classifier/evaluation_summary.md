\# Part 2 — Product Image Classifier Evaluation



\## Dataset and Split



The classifier uses the canonical Fashion-MNIST dataset with 10 product classes.



\- Training set: 55,000 images

\- Validation set: 5,000 images

\- Test set: 10,000 images

\- The test set was kept separate from training and validation.



Each grayscale Fashion-MNIST image is converted to three channels, resized to 224 × 224, and normalized using ImageNet normalization values for compatibility with pretrained ResNet-18.



\## Model and Training Configuration



A pretrained ResNet-18 backbone was used as the feature extractor.



The pretrained convolutional backbone remained frozen during feature extraction, while the final classification head was trained for the 10 Fashion-MNIST classes.



Training configuration:



\- Optimizer: Adam

\- Learning rate: 0.001

\- Batch size: 128

\- Epochs: 10

\- Loss: CrossEntropyLoss



The classifier checkpoint with the best validation accuracy was restored before final test-set evaluation.



\## Fine-Tuning Decision



Feature-extraction validation accuracy reached \*\*90.10%\*\*.



The project uses \*\*80% validation accuracy as the trigger for additional fine-tuning\*\*.



Because 90.10% exceeded the 80% threshold, additional fine-tuning of the pretrained backbone was \*\*not required\*\*.



\- Before fine-tuning validation accuracy: \*\*90.10%\*\*

\- Fine-tuning performed: \*\*No\*\*

\- After fine-tuning validation accuracy: \*\*N/A because fine-tuning was not required\*\*



\## Final Test Accuracy



Final test accuracy:



\*\*88.63%\*\*



This exceeds the required 80% test-accuracy target.



\## 10 × 10 Confusion Matrix



Rows represent actual classes and columns represent predicted classes.



```text

\[\[860,   3,  26,  35,   3,   1,  67,   0,   4,   1],

&#x20;\[  1, 972,   3,  20,   1,   0,   3,   0,   0,   0],

&#x20;\[ 15,   0, 888,  10,  46,   0,  40,   0,   1,   0],

&#x20;\[ 23,   5,  20, 892,  23,   0,  37,   0,   0,   0],

&#x20;\[  1,   0,  86,  39, 789,   0,  82,   0,   3,   0],

&#x20;\[  0,   0,   0,   0,   0, 945,   1,  38,   2,  14],

&#x20;\[160,   0,  67,  40,  94,   0, 633,   0,   5,   1],

&#x20;\[  0,   0,   0,   0,   0,  13,   0, 967,   0,  20],

&#x20;\[  3,   0,   2,   3,   1,   2,  12,   0, 976,   1],

&#x20;\[  0,   0,   0,   0,   1,  12,   0,  45,   1, 941]]

