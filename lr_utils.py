import numpy as np
import h5py

import matplotlib.pyplot as plt
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(x):    #这x是向量
    f=1/(1+np.exp(-x))
    return f

def buildparameter(X):
    n=X.shape[0]
    w=np.zeros(shape=(n,1))
    b=0
    assert (w.shape==(n,1))
    return w,b

def core(w,b,X,Y):
    m=np.shape(X)[1]
    # 正向传播
    Z=np.dot(w.T,X)+b
    A = sigmoid(Z)
    cost= (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))
    # 反向传播
    dZ=A-Y
    dw = (1 / m) * np.dot(X, dZ.T)  #梯度方向
    db = (1 / m) * np.sum(dZ)
    grade={"dw":dw,"db":db}
    return grade,cost

def renewparameter(w,b,X,Y,learning_rate,drop_times):
    n=np.shape(X)[0]
    costs=[]
    for i in range(drop_times):
        grade,cost=core(w,b,X,Y)
        dw=grade["dw"]
        db=grade["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i%100==0:
            costs.append(cost)#每一百个打印一次
            print("迭代的次数:",end="")
            print(i,end="  ")
            print("误差值:",end="")
            print(cost)
    params={"w":w,"b":b}
    grade = {"dw": dw, "db": db}
    return params,grade,costs

def predict(w,b,X):
    m=X.shape[1]
    Y_p=np.zeros(shape=(1,m))
    #w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        Y_p[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_p

def model(train_X, train_Y, test_X, test_Y, num_iterations = 2000, learning_rate = 0.005):
    w,b=buildparameter(train_X)
    parameters,grads,costs=renewparameter(w,b,train_X,train_Y,learning_rate,num_iterations)
    w, b = parameters["w"], parameters["b"]
    Y_prediction_test = predict(w, b, test_X)
    Y_prediction_train = predict(w, b, train_X)
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - train_Y)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - test_Y)) * 100), "%")

if __name__ == '__main__':
    train_set_x_orig, train_Y, test_set_x_orig, test_Y, classes = load_dataset()
    #转换矩阵
    train_X = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T#哇，209行的矩阵（因为训练集里有209张图片），
    test_X = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T                                                               # -1程序你帮我算，最后程序算出来时12288列
    #标准化数据
    train_X = train_X/255
    test_X = test_X/255

    #建立模型
    d = model(train_X, train_Y, test_X, test_Y, num_iterations = 2000, learning_rate = 0.005)
