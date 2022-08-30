import csv
import matplotlib.pyplot as plt
import os
import numpy as np

path = ""
color = ["b","g","r","c","m","y"]
type = ["o","s","^"]
count_color = 0
numb = 0
for i in os.listdir():
    numb = 0

    if ".csv" not in i:
        continue

    if "square" in i:
        numb = 1

    if "triangle" in i:
        numb = 2

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
    plt.xlabel('Patch Size')
    plt.ylabel('Probabilities')
    plt.plot(x,y,f"-{color[count_color%len(color)]}{type[numb]}",label=f"{i}")
    plt.legend(bbox_to_anchor=(1.04,1),loc="upper left",fontsize=5)
    plt.tight_layout()

    #plt.title(f"Results for {i} ")
    count_color += 1
plt.show()






