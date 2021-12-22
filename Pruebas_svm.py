import SVM as SVM

data = SVM.Load_data(r'C:\Users\juann\PycharmProjects\pythonProject1\Aprendizaje\lib\python3\PUJ\Model\SVM\data\input_02.csv')

X_train, y_train, X_test, y_test = SVM.Split_data(data, random_state = 0)


w_f, b_f = SVM.SVM(X_train = X_train, y_train = y_train, n_iter = 300, l = 0.1, lr = 1e-1, epsilon = 1e-4, batch_size = 12, Adam = True, B1 = 0.95, B2 = 0.9999, random_state = 0)
# print(f'w_f: {w_f}')
# print(f'b_f: {b_f}')

y_pred = SVM.Predict(X_test, w_f, b_f)

conf_m, acc = SVM.Confussion_Matrix(y_pred, y_test)
print(f'Accuracy: {acc}')
print('###### Confusion Matrix ######')
print(conf_m)

