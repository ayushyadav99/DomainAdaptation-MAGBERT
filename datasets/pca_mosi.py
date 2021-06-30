import pickle as pkl
from sklearn.decomposition import PCA

def transform(train_data):

    visual_data = train_data[0][0][1].tolist()
    for i in range(1,len(train_data)):
        visual_data.extend(train_data[i][0][1].tolist())

    print(len(visual_data))

    pca = PCA(n_components = 35)
    pca.fit(visual_data)

    for i in range(len(train_data)):
        x_pca = pca.transform(train_data[i][0][1].tolist())
        train_data[i] = list(train_data[i])
        train_data[i][0] = list(train_data[i][0])
        train_data[i][0][1] = x_pca 
        train_data[i][0] = tuple(train_data[i][0])
        train_data[i] =tuple(train_data[i])
    
    return train_data

def main():
    data = pkl.load(open("mosi.pkl", "rb"))

    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']

    train_data = transform(train_data)
    dev_data = transform(dev_data)
    test_data = transform(test_data)

    data = {'train': train_data, 'dev': dev_data, 'test' : test_data}

    with open("mosi_updated.pkl", "wb") as f:
        pkl.dump(data, f)

if __name__=="__main__":
    main()
