w = open('labels.txt',"w")
with open('labels.json') as f:
	content = f.readlines()
	content = [x.strip() for x in content]
	for line in content:
		line = line.split(':',1)[-1]
		line = line.strip()
		line.replace("'","")
		w.write(line)
		w.write('\n')
w.close()