class LabelParser():
	def __init__(self, path="./labels.txt"):
		# path is the path to file with labels.
		self.path = path
		# labels is list of all 1000 labels.
		self.labels = []
	
	def parseLabels(self):
		# creates the labels array from file.
		with open(self.path) as f:
			content = f.readlines()
			content = [x.strip() for x in content]
			for line in content:
				self.labels.append(line)

	def getLabels(self):
		return self.labels