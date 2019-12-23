import numpy as np
import matplotlib.pyplot as plt
name_list = ["IBM TrueNorth \n (128x256)", "NC chip-V1 \n (256x256)",
             "NC chip-V2 \n (512x512)", "NC chip-V3 \n (1024x1024)"]
Toep = [2732, 1800, 588, 236]
Hyb = [552, 106, 39, 17]
x = list(range(len(Toep)))
total_width, n = 0.8, 2
width = total_width / n
plt.figure()
plt.ylabel("Number of Cores")
plt.bar(x, Toep, width=width, label='Toeplitz', fc="g")
#plt.xticks(x, name_list)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, Hyb, width=width, label='Hybrid', fc="r")
for i in range(len(x)):
    x[i] = x[i] - 0.2
plt.xticks(x, name_list)
plt.yscale('log')
plt.legend()


fig = plt.figure()
name_list = ["VGG-16", "REMODEL", "MobileNet", "HFNet-V1",
             "HFNet-V2", "HFNet-V3"]
number_of_cores = [113968, 119172, 6964, 1978, 3814, 4720]
number_of_cores_TN = [908924, 950220, 69960, 31536, 54096, 81128]
accuracy = [71.3, 0, 70.4, 62.46, 67.78, 71.3]
x = list(range(len(number_of_cores)))
total_width, n = 0.8, 2
width = total_width / n
plt.ylabel("Number of Cores")
plt.bar(x, number_of_cores_TN, width=width, label='Core size = 128x256', fc="g")
plt.xticks(x, name_list)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, number_of_cores, width=width, label='Core size = 1024x1024', fc="r")
for i in range(len(x)):
    x[i] = x[i] - 0.2
plt.xticks(x, name_list)
plt.yscale('log')
plt.legend()
plt.show()
