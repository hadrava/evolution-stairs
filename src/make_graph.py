import matplotlib.pyplot as plt
import statistics
import numpy as np

import os
import re

COLLECTION_DATE = 10

def graph_from_data(data, color, label):
   medians = [statistics.median(row) for row in data]
   means = [statistics.mean(row) for row in data]
   maxs = [max(row) for row in data]
   mins = [min(row) for row in data]
   first_quantils = [np.quantile(row, 0.25) for row in data]
   third_quantils = [np.quantile(row, 0.75) for row in data]
   xaxes = [(i+1)*COLLECTION_DATE for i in range(len(data))]
   plt.fill_between(xaxes, mins, maxs, alpha=0.15, color=color)
   plt.fill_between(xaxes,first_quantils, third_quantils, alpha=0.2, color=color)
   plt.plot(xaxes,first_quantils, color=color, linestyle=":")
   plt.plot(xaxes, means, color=color, linewidth=3)
   plt.plot(xaxes, third_quantils, color=color, linestyle=":")
   plt.plot(xaxes, maxs, color=color)
   plt.plot(xaxes, mins, color=color, label=label)
   plt.ylabel("Hodnota fitness")
   plt.ylim(3000,11000)
   plt.grid(True)
   plt.legend()
 
def load_data(directory):
  data =[]
  file_path = os.path.join(directory, "walker_rewards.csv")

  with open(file_path, "r", encoding="utf-8") as in_file:
    read = True
    while read:
      concateneted_rows = []
      for i in range(COLLECTION_DATE):
        line=in_file.readline()
        if line == "":
          read=False
          break
        concateneted_rows.extend([float(i) for i in line.split(";")])
      if len(concateneted_rows) > 0:
        data.append(concateneted_rows)
  return data
 
colors = [(1,0,0), (0,1,0), (0,0,1),
(0.5,0,0), (0,0.5,0), (0,0,0.5),
(0.75,0,0), (0,0.75,0), (0,0,0.75),
(0.25,0,0), (0,0.25,0), (0,0,0.25),
(0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),
(0.75,0.5,0), (0,0.75,0.5), (0.75,0,0.5),
(0.5,0.75,0), (0,0.5,0.75), (0.5,0,0.75),
(0.5,0.25,0), (0,0.5,0.25), (0.5,0,0.25),
(0.25,0.5,0), (0,0.25,0.5), (0.25,0,0.5),
]
pattern = re.compile(r""".*
	c=(?P<crossover>.[a-z]+).*
	cp=(?P<crossover_prob>[0-9.]+).*
	m=(?P<mutation>[a-z]+).*
	mp=(?P<mutation_prob>[0-9.]+).*
	ms=(?P<sigma>[0-9.]+).*""", re.VERBOSE)
MAX = 0
logdirs = os.listdir("logs")
logdirs.sort()
print("\n".join(logdirs))

subgraphs = 2

for i,log in enumerate(logdirs):
  if i % subgraphs == 0:
    plt.xlabel("Číslo generace")
    plt.savefig("test-{}.png".format(i))
    plt.close()
  data = load_data(os.path.join("logs", log))
  logmax = max([max(row) for row in data])
  if logmax > MAX:
     MAX = logmax
     epoch = [i for i,row in enumerate(data) if max(row) == MAX]
     print(logmax, log, epoch)
  print(i)
  plt.subplot(2,1,i%subgraphs+1)
  if i % subgraphs== 0:
    plt.title("Vývoj fitness v závislosti na parametrech evoluce")
  match = pattern.match(str(log))
  label = ("Parameters cp={}, mp={}, ms={}".format(match.group('crossover_prob'), match.group('mutation_prob'), match.group('sigma')))
  graph_from_data(data, color=colors[i % len(colors)], label=label)


plt.savefig("test-{}.png".format(i))
plt.close()

