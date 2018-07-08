from labelParser import LabelParser

classifier = Classifier(categories=['street sign,', 'suit,', 'magpie,', 'jay,', 'sunglasses,'])
image = classifier.readImage('../1.jpg')
image = classifier.prepare(image)

probability = classifier.getPredictionProbability(image)
print(probability)

print(classifier.getTopClassLabel(probability))
print(classifier.getTopKProbabilities(probability, 5))
print(classifier.getTopKClassLabels(probability, 5))
print(classifier.getCategoryProbabilities(probability))