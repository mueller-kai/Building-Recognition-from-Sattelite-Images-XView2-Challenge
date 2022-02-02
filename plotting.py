import matplotlib. as plt

preformance = [0.5944, 0.6099, 0.6167, 0.6141, 0.6003, 0.5424]

plt.style.use("ggplot")
plt.figure()
plt.plot(preformance, label="f1")
plt.title("f1 score for threshholds")
plt.xlabel("Threshhold", )
plt.ylabel("performance")
plt.legend(loc="lower left")
plt.savefig(f'f1.png')