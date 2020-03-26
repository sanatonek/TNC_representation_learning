

def create_simulated_dataset(path='./data/simulated_data/'):
    if not os.listdir(path):
        raise ValueError('Data does not exist')
    x_train = pickle.load(open(os.path.join(path, 'x_train.pkl'), 'wb'))
    x_test = pickl.load(open(os.path.join(path, 'x_test.pkl'), 'wb'))
    y_train = pickl.load(open(os.path.join(path, 'state_train.pkl'), 'wb'))
    y_test = pickle.load(open(os.path.join(path, 'state_test.pkl'), 'wb'))