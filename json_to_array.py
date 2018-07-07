import json

w = open('labels.txt',"w")
with open('labels.json') as f:
	content = f.readlines()
	content = [x.strip() for x in content]
	for line in content:
		line = line.split(':',1)[-1]
		line = line.strip()
		w.write(line)
w.close()