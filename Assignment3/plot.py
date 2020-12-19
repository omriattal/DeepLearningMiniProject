import matplotlib.pyplot as plt
import re

# x = training steps
# y = loss value

text = open('rnn_10_1000', 'r').read()

training_steps = []
loss_values = []

for number in re.findall(r'Step=(\d+)', text):
    training_steps.append(int(number))

for number in re.findall(r'Loss=(\d+.\d+)', text):
    loss_values.append(float(number))

print(training_steps)
print(loss_values)

plt.plot(training_steps, loss_values)
plt.xlabel('Training Step')
plt.ylabel('Loss Value')
plt.show()
