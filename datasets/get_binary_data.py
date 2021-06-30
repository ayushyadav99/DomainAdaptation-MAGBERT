from statistics import median 
import pickle as pkl

f_mosei = open('mosi.pkl', 'rb')
mosei_data = pkl.load(f_mosei)
mosei_train_data = mosei_data['train']
mosei_test_data = mosei_data['test']
mosei_dev_data = mosei_data['dev']

sentiment_values = []
for d in range(len(mosei_train_data)):
    sentiment_values.append(float(mosei_train_data[d][1][0][0]))
for d in range(len(mosei_test_data)):
    sentiment_values.append(float(mosei_test_data[d][1][0][0]))
for d in range(len(mosei_dev_data)):
    sentiment_values.append(float(mosei_dev_data[d][1][0][0]))

mid_val = median(sentiment_values)
print("mid", mid_val)

f_new = open('mosi_bin.pkl', 'wb')

count_1 = 0
count_0 = 0
for d in range(len(mosei_train_data)):
    val = 0 if float(mosei_train_data[d][1][0][0])<=mid_val else 1
    mosei_train_data[d][1][0][0] = val
    if val == 0:
        count_0+=1
    else:
        count_1 +=1
print(count_0, count_1)

count_1 = 0
count_0 = 0
for d in range(len(mosei_test_data)):
    val = 0 if float(mosei_test_data[d][1][0][0])<=mid_val else 1
    mosei_test_data[d][1][0][0] = val
    if val == 0:
        count_0+=1
    else:
        count_1 +=1
print(count_0, count_1)

count_1 = 0
count_0 = 0
for d in range(len(mosei_dev_data)):
    val = 0 if float(mosei_dev_data[d][1][0][0])<=mid_val else 1
    mosei_dev_data[d][1][0][0] = val
    if val == 0:
        count_0+=1
    else:
        count_1 +=1
print(count_0, count_1)
new_data = dict({'train': mosei_train_data, 'test': mosei_test_data, 'dev': mosei_dev_data})
pkl.dump(new_data, f_new)
