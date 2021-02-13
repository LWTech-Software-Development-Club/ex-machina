import pandas as pd

# ===============================================================================
# Get the Training Data
# ===============================================================================

def getSanitizedDataFromTrainingFile(fileName):
	print("Loading Training File: " + fileName)
	data = pd.read_csv(fileName, sep="\t", header=None)
	data_clean = pd.DataFrame(data)
	data_clean.columns = ['label', 'text']
	data_clean['text'] = sanitizeText(data_clean['text'])
	return data_clean

def getTestFile(fileName):
	print("Loading Test File: " + fileName)
	data = pd.read_csv(fileName, sep="\t", header=None)
	data_clean = pd.DataFrame(data)
	data_clean.columns = ['text']
	data_clean['text'] = sanitizeText(data_clean['text'])
	return data_clean

def sanitizeText(text):
	print("Sanitizing text...")
	text = text.str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()
	text = text.str.lower()
	text = text.str.split()
	return text

print('Commencing training file input process.')

training_data = getSanitizedDataFromTrainingFile('SMSSpamCollection.short')
# TODO: Get it to append multiple files to the same massive training data clean var
#training_data &= getSanitizedDataFromTrainingFile('SMSSpamCollection2.short')

print('Training files loaded and sanitized.')


# Old Code:
# print(data_clean) # Debug
# Previously, it would split the training data into two groups, this is no longer the desired behavior.
# train_data = data_clean.sample(frac=0.8, random_state=1).reset_index(drop=True)
# test_data = data_clean.drop(train_data.index).reset_index(drop=True) # drop everything from the index point which doesn't move from where it left off
# train_data = train_data.reset_index(drop=True)
# print(train_data)
# print(test_data)


# ===============================================================================
# Train the Data
# ===============================================================================

print('Training the data, this might take a while...')

# Create frequency table
vocabulary = list(set(training_data['text'].sum()))
word_counts_per_sms = pd.DataFrame([
	[row[1].count(word) for word in vocabulary]
	for _, row in training_data.iterrows()], columns=vocabulary)
training_data = pd.concat([training_data.reset_index(), word_counts_per_sms], axis=1).iloc[:,1:]
# print(train_data.style)
# print(tabulate(train_data, headers='keys', tablefmt='psql'))
probabilityOfSpam = training_data['label'].value_counts()['spam'] / training_data.shape[0]
probabilityOfHam = training_data['label'].value_counts()['ham'] / training_data.shape[0]
numberSpam = training_data.loc[training_data['label'] == 'spam', 'text'].apply(len).sum()
numberHam = training_data.loc[training_data['label'] == 'ham', 'text'].apply(len).sum()
vocabularySize = len(training_data.columns) - 3
# print(test_data)

print('Data training complete.')

# ===============================================================================
# Test the Trained Data against Real Messages
# ===============================================================================

def probabilityWordIsSpam(word):
	# Alpha — the coefficient for the cases when a word in the message is absent in our dataset.
	alpha = 1
	if word in training_data.columns:
		return (training_data.loc[training_data['label'] == 'spam', word].sum() + alpha) / (numberSpam + alpha * vocabularySize)
	else:
		return 1

def probabilityWordIsHam(word):
	# Alpha — the coefficient for the cases when a word in the message is absent in our dataset.
	alpha = 1
	if word in training_data.columns:
		return (training_data.loc[training_data['label'] == 'ham', word].sum() + alpha) / (numberHam + alpha * vocabularySize)
	else:
		return 1

def classify(message):
	p_spam_given_message = probabilityOfSpam
	p_ham_given_message = probabilityOfHam
	for word in message:
		p_spam_given_message *= probabilityWordIsSpam(word)
		p_ham_given_message *= probabilityWordIsHam(word)
	if p_ham_given_message > p_spam_given_message:
		return 'ham'
	elif p_ham_given_message < p_spam_given_message:
		return 'spam'
	else:
		return 'needs human classification'

def listToString(s):
	return ' '.join(s)

print('Getting real data to test against the trained data.')

test_data = getTestFile('real_messages.txt')

print('Testing the trained data against the real data.')

for _, row in test_data.iterrows():
	print(classify(row['text']) + " | MSG: " + listToString(row['text']))

print('Program gracefully terminated.')
