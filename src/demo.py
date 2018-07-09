from labelParser import LabelParser

classifier = Classifier(categories=['street sign,', 'suit,', 'magpie,', 'jay,', 'sunglasses,'])
image = classifier.readImage('../1.jpg')
image = classifier.prepare(image)

probability = classifier.getPredictionProbability(image)

# Print all probabilities.
print(probability)

# Print predicted class.
print(classifier.getTopClassLabel(probability))

# Print 5 highest probabilities.
print(classifier.getTopKProbabilities(probability, 5))

# Print top 5 predicted classes.
print(classifier.getTopKClassLabels(probability, 5))

# Print probabilities of category labels.
print(classifier.getCategoryProbabilities(probability))