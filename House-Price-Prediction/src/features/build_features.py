from sklearn.model_selection import train_test_split
def Train_Split(datas):
    # seperate input features in x
    x = datas.drop('price', axis=1)
    # store the target variable in y
    y = datas['price']
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)
    print("Test Train Split: ")
    print( x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test