import csv
from keras.models import Sequential
from keras.layers import Dense, Flatten
from matplotlib import pyplot

inputs = []
outputs = []

with open("testdb.csv", "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        inputs.append([float(x) for x in row[0:-2]])
        outputs.append([float(x) for x in row[-2:]])

test_input = []
test_output = []

for i in range(1000):
    test_input.append(inputs.pop(0))
    test_output.append(outputs.pop(0))

# 1. define the network
model = Sequential()
model.add(Flatten(input_shape=(12, )))
model.add(Dense(25, activation='relu'))
model.add(Dense(2, activation='relu'))

# 2. compile the network
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 3. fit the network
history = model.fit(inputs, outputs, epochs=30, batch_size=30)

pyplot.plot(history.history["mae"])
pyplot.show()

fully_correct = 0
right_winner = 0
for i in range(1000):
    output = model.predict([test_input[i]]).tolist()[0]
    output = [int(x) for x in output]
    expected = [int(x) for x in test_output[i]]

    if output == expected:
        fully_correct += 1
        print("Correct!")
    else:
        print("Wrong!")

    if output[0] > output[1] and expected[0] > expected[1]:
        right_winner += 1
    elif output[0] < output[1] and expected[0] < expected[1]:
        right_winner += 1
    elif output[0] == output[1] and expected[0] == expected[1]:
        right_winner += 1

full_accuracy = 100 * fully_correct / 1000
print(f"Fully Correct Accuracy: {full_accuracy}%")
winner_accuracy = 100 * right_winner / 1000
print(f"Winner Accuracy: {winner_accuracy}%")
