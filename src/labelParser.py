class LabelParser():
	def __init__(self, path):
		self.path = path
		self.labels = []
	
	def getLabels(self):
		with open(self.path) as f:
			content = f.readlines()
			content = [x.strip() for x in content]
			for lines in content:
				self.labels.add(line)
		return self.labels