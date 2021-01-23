from tabulate import tabulate
import pandas as pd

data = pd.read_csv('SMSSpamCollection', sep="\t", header=None)
sms_data_clean = pd.DataFrame(data)
sms_data_clean.columns = ['Label', 'SMS']

sms_data_clean['SMS'] = sms_data_clean['SMS'].str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()
sms_data_clean['SMS'] = sms_data_clean['SMS'].str.lower()
sms_data_clean['SMS'] = sms_data_clean['SMS'].str.split()

print(sms_data_clean)

# split and train test data
train_data = sms_data_clean.sample(frac=0.8, random_state=1).reset_index(drop=True)
test_data = sms_data_clean.drop(train_data.index).reset_index(drop=True) # drop everything from the index point which doesn't move from where it left off
train_data = train_data.reset_index(drop=True)

print(train_data)
print(test_data)

# Make massive frequency table

vocabulary = list(set(train_data['SMS'].sum()))
word_counts_per_sms = pd.DataFrame([
    [row[1].count(word) for word in vocabulary]
    for _, row in train_data.iterrows()], columns=vocabulary)
train_data = pd.concat([train_data.reset_index(), word_counts_per_sms], axis=1).iloc[:,1:]

print(train_data.style)

# print(tabulate(train_data, headers='keys', tablefmt='psql'))
