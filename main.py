import pandas as pd

# Preparing data input from text files

# ===============================================================================
# Get the Data
# ===============================================================================

def getTrainingFile(fileName):
	data = pd.read_csv(fileName, sep="\t", header=None)
	data_clean = pd.DataFrame(data)
	data_clean.columns = ['label', 'text']
	data_clean['text'] = sanitizeText(data_clean['text'])
	return data_clean

def getTestFile(fileName):
	data = pd.read_csv(fileName, sep="\t", header=None)
	data_clean = pd.DataFrame(data)
	data_clean.columns = ['text']
	data_clean['text'] = sanitizeText(data_clean['text'])
	return data_clean

def sanitizeText(text):
	text = text.str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()
	text = text.str.lower()
	text = text.str.split()
	return text

data_clean = getTrainingFile('SMSSpamCollection.short')
# print(data_clean)
# split and train test data
train_data = data_clean.sample(frac=0.8, random_state=1).reset_index(drop=True)
test_data = data_clean.drop(train_data.index).reset_index(drop=True) # drop everything from the index point which doesn't move from where it left off
train_data = train_data.reset_index(drop=True)
# print(train_data)
# print(test_data)

# ===============================================================================
# Train the Data
# ===============================================================================

def probabilityWordIsSpam(word):
	# Alpha — the coefficient for the cases when a word in the message is absent in our dataset.
	alpha = 1
	if word in train_data.columns:
		return (train_data.loc[train_data['label'] == 'spam', word].sum() + alpha) / (numberSpam + alpha * vocabularySize)
	else:
		return 1

def probabilityWordIsHam(word):
	# Alpha — the coefficient for the cases when a word in the message is absent in our dataset.
	alpha = 1
	if word in train_data.columns:
		return (train_data.loc[train_data['label'] == 'ham', word].sum() + alpha) / (numberHam + alpha * vocabularySize)
	else:
		return 1

# Make massive frequency table
vocabulary = list(set(train_data['text'].sum()))
word_counts_per_sms = pd.DataFrame([
	[row[1].count(word) for word in vocabulary]
	for _, row in train_data.iterrows()], columns=vocabulary)
train_data = pd.concat([train_data.reset_index(), word_counts_per_sms], axis=1).iloc[:,1:]
# print(train_data.style)
# print(tabulate(train_data, headers='keys', tablefmt='psql'))
probabilityOfSpam = train_data['label'].value_counts()['spam'] / train_data.shape[0]
probabilityOfHam = train_data['label'].value_counts()['ham'] / train_data.shape[0]
numberSpam = train_data.loc[train_data['label'] == 'spam', 'text'].apply(len).sum()
numberHam = train_data.loc[train_data['label'] == 'ham', 'text'].apply(len).sum()
vocabularySize = len(train_data.columns) - 3
# print(test_data)

# ===============================================================================
# Train the Data
# ===============================================================================

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

getTestFile('real_messages.txt')

for _, row in data_clean.iterrows():
	print(classify(row['text']) + " | MSG: " + listToString(row['text']))

