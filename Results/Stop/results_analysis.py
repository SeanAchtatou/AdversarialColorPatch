import csv
import matplotlib.pyplot as plt
import os
import numpy as np

path = ""

for i in os.listdir():
    if ".csv" not in i:
        continue

    x = []
    y = []
    labels = []
    with open(f"{i}","r") as f:
        csv_read = csv.reader(f)
        count = 0
        for j in csv_read:
            if count == 0:
                count += 1
                pass
            else:
                x.append(float(j[0]))
                y.append(float(j[5]))
                labels.append([float(j[0]),float(j[5]),float(j[-1])])

    plt.xlim(max(x),min(x))
    plt.ylim(0,1)
    plt.xlabel('Patch Size based on % of the image')
    plt.ylabel('Probability of Highest Class')
    plt.plot(x,y,"-ks",label=f"{i}")
    plt.legend(loc="upper center",fontsize=5)

    plt.show()





