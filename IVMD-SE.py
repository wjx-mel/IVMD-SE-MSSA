import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Input, Model, Sequential
from SE import cal_fluctuation_dispersion_entropy
import time
begintime = time.time()

# 进行训练和预测
imf = np.load('结果/15朝阳广场进站IVMD-data.npz')['imf']
test_pred_each_mode, test_real_each_mode = [], []
a = ['预测值6.csv', '预测值7.csv', '预测值8.csv', '预测值9.csv', '预测值10.csv', '预测值11.csv', '预测值12.csv', '预测值13.csv',
     '预测值14.csv', '预测值15.csv', '预测值16.csv', '预测值17.csv', '预测值18.csv', '预测值19.csv', '预测值20.csv']


def attention_3d_block(inputs, timesteps):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(timesteps, activation='linear')(a)
    a_probs = Permute((2, 1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


for mode in range(imf.shape[0]):

    print('对第', mode + 1, '个VMD分量进行_LSTM回归建模')
    data = imf[mode, :].reshape(-1, 1)
    se = cal_fluctuation_dispersion_entropy(data, 1, 3, 6)
    time_len = data.shape[0]
    train_rate = 0.746
    seq_len = 5
    pre_len = 3
    train_data, train_label, test_data, test_label = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
    train_data = np.array(train_data)
    train_data = np.reshape(train_data, [-1, seq_len])
    train_label = np.array(train_label)
    train_label = np.reshape(train_label, [-1, pre_len])
    test_data = np.array(test_data)
    test_data = np.reshape(test_data, [-1, seq_len])
    test_label = np.array(test_label)
    test_label = np.reshape(test_label, [-1, pre_len])

    ss_X = MinMaxScaler(feature_range=(0, 1)).fit(train_data.astype(np.float))
    ss_Y = MinMaxScaler(feature_range=(0, 1)).fit(train_label.astype(np.float))
    if se <= 0.8:
        train_data = ss_X.transform(train_data)
        train_label = ss_Y.transform(train_label)

        test_data = ss_X.transform(test_data)
        test_label = ss_Y.transform(test_label)
        model = Sequential()
        model.add(Dense(units=64, input_dim=5, activation='linear', use_bias=True, kernel_initializer='uniform'))
        model.add(Dense(units=3, activation='linear', kernel_initializer='uniform'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        history = model.fit(train_data, train_label, epochs=250, validation_data=(test_data, test_label),
                            batch_size=16, verbose=1)
        model.save('model/3步金湖广场IVMD-BP-imf' + str(mode + 1) + 'model.h5')
        plt.figure()
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.plot(history.history['loss'], label='training')
        plt.plot(history.history['val_loss'], label='testing')
        plt.title('loss curve')
        plt.legend()
        test_pred = model.predict(test_data)
    else:
        train_data = ss_X.transform(train_data).reshape(train_data.shape[0], seq_len, -1)
        train_label = ss_Y.transform(train_label)

        test_data = ss_X.transform(test_data).reshape(test_data.shape[0], seq_len, -1)
        test_label = ss_Y.transform(test_label)
        num_epochs = 250  # 迭代次数
        lr = 0.001  # 学习率
        sequence, feature = train_data.shape[-2:]
        output_node = train_label.shape[1]
        #  利用优化的参数进行建模
        tf.random.set_seed(0)
        i = Input(batch_shape=(None, sequence, 1))
        o = LSTM(128, input_shape=(12, 1), return_sequences=True)(i)
        o = LSTM(64, return_sequences=True)(o)
        o = attention_3d_block(o, 5)
        o = Dropout(0.2)(o)
        o = Flatten()(o)
        o = Dense(output_node, activation='sigmoid')(o)
        model = Model(inputs=[i], outputs=[o])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
        # 训练模型
        history = model.fit(train_data, train_label, epochs=num_epochs, validation_data=(test_data, test_label),
                            batch_size=16, verbose=1)
        model.save('model/3步朝阳广场IVMD-ALSTM_imf' + str(mode + 1) + 'model.h5')
        # 画loss曲线
        plt.figure()
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.plot(history.history['loss'], label='training')
        plt.plot(history.history['val_loss'], label='testing')
        plt.title('loss curve')
        plt.legend()
        test_pred = model.predict(test_data)

    # 对测试结果进行反归一化
    test_label1 = ss_Y.inverse_transform(test_label)
    test_pred1 = ss_Y.inverse_transform(test_pred)
    dataframe = pd.DataFrame(test_pred1)
    dataframe.to_csv(a[mode])
    test_pred_each_mode.append(test_pred1)
    test_real_each_mode.append(test_label1)

# In[] 叠加预测结果
print('----叠加预测结果-------')
totaltime = time.time()-begintime
print(totaltime)

test_pred = np.array(test_pred_each_mode).reshape(imf.shape[0], -1).sum(axis=0).reshape(-1, 3)

dataframe = pd.DataFrame(test_pred)
dataframe.to_csv('预测值1.csv')

data1 = np.load('结果/15朝阳广场进站IVMD-data.npz')['data'].reshape(-1, 1)
time_len = data1.shape[0]
train_rate = 0.746
seq_len = 5
pre_len = 3
train_data2, train_label2, test_data2, test_label2 = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)
test_label2 = np.array(test_label2)
test_label2 = np.reshape(test_label2, [-1, pre_len])

# 各分量画图
plt.figure()
for i in range(imf.shape[0]):
    plt.subplot(3, np.ceil(imf.shape[0] / 3), i + 1)
    plt.plot(test_real_each_mode[i], c='r', label='true')
    plt.plot(test_pred_each_mode[i], c='b', label='predict')
    plt.title('IMF' + str(i + 1))
    plt.legend()
plt.show()

# In[]
# 画出测试集的值
plt.figure()
plt.plot(test_label2, c='r', label='true')
plt.plot(test_pred, c='b', label='predict')
plt.legend()
plt.show()

np.savez('结果/3步朝阳广场IVMD-ALSTM-result.npz', true=test_label2, pred=test_pred)

print("RMSE:")
for i in range(pre_len):
    actual = [row[i] for row in test_label2]
    predicted = [predict[i] for predict in test_pred]
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(np.mean(np.square(predicted - actual)))
    print('t+%d RMSE: %f' % ((i + 1), rmse))

print("MAE:")
for i in range(pre_len):
    actual = [row[i] for row in test_label2]
    predicted = [predict[i] for predict in test_pred]
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = np.mean(np.abs(predicted - actual))
    print('t+%d MAE: %f' % ((i + 1), mae))


print("MAPE:")
for i in range(pre_len):
    actual = [row[i] for row in test_label2]
    predicted = [predict[i] for predict in test_pred]
    actual = np.array(actual)
    predicted = np.array(predicted)
    MAPE = np.mean(np.abs((predicted - actual) / actual))
    print('t+%d MAPE: %f' % ((i + 1), MAPE))


print("SDE:")
for i in range(pre_len):
    actual = [row[i] for row in test_label2]
    predicted = [predict[i] for predict in test_pred]
    actual = np.array(actual)
    predicted = np.array(predicted)
    SDE = np.sqrt(np.mean(np.square((actual - predicted)-np.mean(actual - predicted))))
    print('t+%d SDE: %f' % ((i + 1), SDE))




