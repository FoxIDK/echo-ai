# Imports.
import json
import nltk
import numpy as np
import os
import pickle
import random
import sys
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Ftrl
from nltk.stem import WordNetLemmatizer


# Pre-run.
os.system("clear")
sys.tracebacklimit = 0

try:
	epv = int(input("\n[?] How many epochs do you want to cycle?: "))
	print("\nAvailable models: Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Ftrl")
	tmv = input("[?] Which model do you want to use?: ").lower()

	directory = "intenses_db/"
	data = []
	for filename in os.listdir(directory):
		if filename.endswith(".json"):
			with open(directory + filename, "r") as f:
				file_data = json.load(f)
			data.append(file_data)
	with open(directory + "training-data.json", "w") as f:
		json.dump(data, f, indent=4)
	os.system("rm ./neural_data/training-data.json")
	os.system("mv ./intenses_db/training-data.json ./neural_data/training-data.txt")

	with open('./neural_data/training-data.txt', 'r') as f:
		file_contents = f.close()
	with open('./neural_data/training-data.txt', 'r') as file:
		data = file.read()
	data = data.replace('        ]\n    },\n    {\n        "intents": [', '            ,')
	with open('./neural_data/training-data.txt', 'w') as file:
		file.write(data)
	with open("./neural_data/training-data.txt", "r") as f:
		lines = f.readlines()
	lines = lines[1:-1]
	with open("./neural_data/training-data.txt", "w") as f:
		f.writelines(lines)

	os.system("mv ./neural_data/training-data.txt ./neural_data/training-data.json")

	lemmatizer = WordNetLemmatizer()
	intents = json.loads(open("./neural_data/training-data.json").read())

	# Creating empty lists to store data
	words = []
	classes = []
	documents = []
	ignore_letters = ["?", "!", ".", ","]
	for intent in intents['intents']:
		for pattern in intent['patterns']:
			# Separating words from patterns
			word_list = nltk.word_tokenize(pattern)
			words.extend(word_list)
			documents.append((word_list, intent['tag']))

			if intent['tag'] not in classes:
				classes.append(intent['tag'])

	words = [lemmatizer.lemmatize(word)
			for word in words if word not in ignore_letters]
	words = sorted(set(words))

	# Saving the words and classes list to binary files
	pickle.dump(words, open('./neural_data/words.pkl', 'wb'))
	pickle.dump(classes, open('./neural_data/classes.pkl', 'wb'))

	training = []
	output_empty = [0]*len(classes)
	for document in documents:
		bag = []
		word_patterns = document[0]
		word_patterns = [lemmatizer.lemmatize(
			word.lower()) for word in word_patterns]
		for word in words:
			bag.append(1) if word in word_patterns else bag.append(0)

		# Making a copy of the output_empty
		output_row = list(output_empty)
		output_row[classes.index(document[1])] = 1
		training.append([bag, output_row])
	random.shuffle(training)
	training = np.array(training, dtype=object)

	# Splitting the data
	train_x = list(training[:, 0])
	train_y = list(training[:, 1])

	model = Sequential()
	model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(len(train_y[0]), activation='softmax'))

	tensorboard_callback = TensorBoard(log_dir='./tlogs', histogram_freq=1, write_graph=True, write_images=True)

	if tmv == "sgd":
		optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
	elif tmv == "adam":
		optimizer = Adam(learning_rate=0.001)
	elif tmv == "rmsprop":
		optimizer = RMSprop(learning_rate=0.001)
	elif tmv == "adagrad":
		optimizer = Adagrad(learning_rate=0.001)
	elif tmv == "adadelta":
		optimizer = Adadelta(learning_rate=1.0, rho=0.95)
	elif tmv == "adamax":
		optimizer = Adamax(learning_rate=0.001)
	elif tmv == "nadam":
		optimizer = Nadam(learning_rate=0.001)
	elif tmv == "ftrl":
		optimizer = Ftrl(learning_rate=0.001)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	hist = model.fit(np.array(train_x), np.array(train_y), epochs=epv, batch_size=5, verbose=1, callbacks=[tensorboard_callback])

	model.save("./neural_data/echo.h5", hist)

	print("\nTraining session complete!\n")
	os.system("tensorboard --logdir=./tlogs --port=6006")

except json.decoder.JSONDecodeError as e:
	print("\n[!] Error decoding JSON or training data:", e)