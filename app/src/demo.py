from labelParser import LabelParser
from classifier import Classifier
import numpy as np
classifier = Classifier(categories=['street sign,', 'suit,', 'magpie,', 'jay,', 'sunglasses,'], labelsPath='../labels.txt')
image = classifier.readImage('../eggnog.jpg')
image = classifier.prepare(image)

energy = classifier.getEnergy(image)
print(np.argmax(energy))

#probability = classifier.getPredictionProbability(image)

# Print all probabilities.
#print(probability)

# Print predicted class.
#print(classifier.getTopClassLabel(probability))

# Print 5 highest probabilities.
#print(classifier.getTopKProbabilities(probability, 5))

# Print top 5 predicted classes.
#print(classifier.getTopKClassLabels(probability, 5))

# Print probabilities of category labels.
#print(classifier.getCategoryProbabilities(probability))
