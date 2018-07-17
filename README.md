MyTorch Processing
==================

### Processing code for mytorch cloud machine learning framework.

Currently, the code can be used to predict labels on any given image.
These labels can be any of the 1000 labels from the Imagenet dataset.
Training specific model on specific labels will come out later.

## How to Use

Create an instance of class Classifier as follows:
```
classifier = Classifier(categories, model, labelsPath)
```
Here, categories is any subset of the 1000 labels on Imagenet, whose presence/
absence the user is concerned with in any image.

The classifier itself can be used to read and prepare an image before classification.

```
image = classifier.readImage(imagePath)
image = classifier.prepare(image)
```

Probabilities for each of the 1000 classes are obtained by:
```
probabilies = classifier.getPredictionProbability(image)
```

Various different stats can be obtained from this list of probabilities,
such as:

* getEnergy(image) - model output just before going to softmax
* getTopProbability(probabilities) - get highest probability from list of probabilities
* getTopProbabilityIndex(probabilities) - get index of class with highest probability
* getTopClassLabel(probabilities) - get label of class with highest probability
* getTopKProbabilities(probabilities) - get 'k' top predicted probabilities
* get TopKProbabilityIndices(probabilities) - get indices of classes with 'k' highest probabilities
* getgetTopKClassLabels(probabilities) - get labels of classes with 'k' highest probabilities
* getCategoryProbabilities(probabilities) - get prediction probabilities for each of the interested category labels.

Code demo available at src/demo.py

Run unit tests by running:
```
python -m tests.classifierTest
```