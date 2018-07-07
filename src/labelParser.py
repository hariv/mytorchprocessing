class LabelParser():
	def __init__(self, path):
		self.path = path
		self.labels = []
	
	def parseLabels(self):
		with open(self.path) as f:
			content = f.readlines()
			content = [x.strip() for x in content]
			for line in content:
				self.labels.append(line)

	def getLabels(self):
		return self.labels