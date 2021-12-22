# TODO: recibir datos --- Separar X , y

import numpy, os, random, sys, pandas, math
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), '../../lib/python3'))


def Load_data(filename):
    data = pandas.read_csv(filename, delimiter=',', header = None)
    return data


def Center(data):
    X = data.iloc[:, 0:data.shape[1] - 1]
    m = X.mean(axis=0)
    return X - m

def Flatten( array ):
    f = None
    for l in range( len( array ) ):
      if f is None:
        f = array[ l ].flatten( )
      else:
        f = numpy.append( f,array[ l ].flatten( ) )
      # end if

    return f



def Split_data(data, test_size=0.2, random_state = 42):
    data = data.sample(frac=1, random_state = random_state).reset_index(drop=True)
    X = Center(data)
    y = data.iloc[:, -1]
    y.iloc[numpy.where(y < 1)] = -1
    X_train = X.iloc[0:int(X.shape[0] * (1 - test_size)), :]
    X_test = X.iloc[int(X.shape[0] * (1 - test_size)):, :]
    y_train = y.iloc[0:int(X.shape[0] * (1 - test_size))]
    y_test = y.iloc[int(X.shape[0] * (1 - test_size)):]

    return X_train, y_train, X_test, y_test



# TODO: Inicializar pesos ---
def Initialize(X, random_state =42):
    random.seed(random_state)
    w = numpy.random.uniform(low=-1, high=1, size=(X.shape[1], 1))
    b = numpy.random.uniform(low=-1, high=1, size=(1,))
    return w, b



# TODO: Debug ---
def Init_Debug():
    x = []
    y = []

    plt.ion()
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    line, = ax.plot(x, y, '-r')
    plt.show()
    return x, y,  ax, line

def Debug_function(ax , x, y, line, n, J):
    x.append(n)
    y.append(J)
    ax.set_xlim([0, n + 10])
    ax.set_ylim([0, max(y) + 2])
    ax.set_title('Debugging')
    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Funcion de Costo J')
    line.set_data(x, y)
    plt.pause(0.1)

def Predict(X, w, b):
    y_pred = numpy.dot(X, w) - b
    return y_pred

# TODO: Crear funcion de costo Bisagra ---

def J_Bisagra(X_train, y_train, w, b, l=1, derivate=False):

    y_pred = Predict(X_train,w,b)
    L = []

    if not derivate:
        for i in range(len(y_train)):
            if y_train[i] * y_pred[i] >= 1:
                L.append(0)
            else:
                L.append(1 - y_train[i] * y_pred[i])

        return (sum(L) / len(L)) + (l * ((numpy.linalg.norm(w) ** 2) * (float(b) ** 2)))
    if derivate:
        Jw = numpy.empty(shape = [1 ,len(w)] )
        Jw = Jw.reshape(-1, len(w))
        Jb = []
        for i in range(len(y_train)):
            if y_train[i] * y_pred[i] >= 1:
                Jw = numpy.append(Jw, numpy.zeros(shape=(1, len( w ))))
                Jb.append(0)

            else:

                Jw = numpy.append(Jw, (-y_pred[i] * (X_train.iloc[i, :]).values.flatten())) ##### Seguir ######
                #Jw = numpy.append(Jw, (-y_pred[i] * x_res))
                Jb.append(y_pred[i] * b)


        Jw = Jw.reshape(-1, len(w))
        Jwf = (Jw.mean(axis = 0)).reshape(-1, 1) + 2 * l * w
        Jbf = (sum(Jb) / len(Jb)) + 2 * l * b

        return Jwf, Jbf



def SVM(X_train, y_train, n_iter=100, epsilon=1e-4, lr=1e-3, l=1, Debuging = True, batch_size = 1, Adam =False, B1 = 0.9, B2 = 0.999, random_state = None):
    J_i = 0
    delta_J = 1
    J = 10
    w, b = Initialize(X_train, random_state= random_state)
    n = 0
    t = 1
    theta = numpy.insert(w, len(w), b, axis =0)
    m = numpy.zeros(shape=(len(w) + 1, 1))
    v = numpy.zeros(shape=(len(w) + 1, 1))

    if Debuging:
        x, y, ax, line = Init_Debug()
    condition = True

    def Adam_Op(Jw, Jb, m, v, t, w, b):
        theta = numpy.insert(w, len(w), b, axis=0)
        e = 1e-8
        G = numpy.insert(Jw, len(Jw), Jb, axis=0)
        m = B1 * m + (1 - B1) * G
        v = B2 + v + (1 - B2) * (numpy.square(G))
        m_c = m / (1 - B1 ** t)
        v_c = v / (1 - B2 ** t)

        theta += lr * numpy.divide(m_c, (numpy.sqrt(v_c) + e))

        w = theta[:len(w)]
        b = theta[-1]

        t += 1
        return w, b, t


    while condition:
        if batch_size == 1:
            J = J_Bisagra(X_train, y_train, w, b, derivate=False)
            if n > 0:
                delta_J = J - J_i

            J_i = J
            Jw, Jb = J_Bisagra(X_train, y_train, w, b, l, derivate=True)

            if Adam:
                w, b, t = Adam_Op(Jw, Jb, m, v, t, w, b)

            else:
                w += lr * Jw.reshape(-1, 1)
                b += lr * Jb
            n += 1
            if Debuging:
                Debug_function(ax, x, y, line, n, J)


            if n >= n_iter:
                condition = False
            if abs(delta_J) < epsilon:
                condition = False
        else:
            num_batch = math.ceil(X_train.shape[0]/batch_size)

            for i in range(num_batch):

                if i < num_batch - 2:
                    X_train_b = X_train.iloc[i * batch_size:(i + 1) * batch_size, :].reset_index(drop=True)
                    y_train_b = y_train[i * batch_size:(i + 1) * batch_size].reset_index(drop=True)

                    J = J_Bisagra(X_train_b, y_train_b, w, b, derivate=False)

                    if n > 0:
                        delta_J = J - J_i

                    J_i = J
                    Jw, Jb = J_Bisagra(X_train_b, y_train_b, w, b, l, derivate=True)

                    if Adam:
                        w, b, t = Adam_Op(Jw, Jb, m, v, t, w, b)

                    else:
                        w += lr * Jw.reshape(-1, 1)
                        b += lr * Jb

                else:
                    X_train_b = X_train.iloc[i * batch_size:].reset_index(drop=True)
                    y_train_b = y_train[i * batch_size:].reset_index(drop=True)
                    J = J_Bisagra(X_train_b, y_train_b, w, b, derivate=False)
                    if n > 0:
                        delta_J = J - J_i

                    J_i = J
                    Jw, Jb = J_Bisagra(X_train_b, y_train_b, w, b, l, derivate=True)

                    if Adam:
                        w, b, t = Adam_Op(Jw, Jb, m, v, t, w, b)
                    else:
                        w += lr * Jw.reshape(-1, 1)
                        b += lr * Jb


                if n >= n_iter:
                    condition = False
                if J <1 and abs(delta_J) < epsilon:
                    condition = False
            n += 1
            if Debuging:
                Debug_function(ax, x, y, line, n, J)




    print(f'Numero de Iteraciones: {n}')
    print(f'Delta_J final: {delta_J}')
    print(f'J final: {J}')
    return w, b



def Confussion_Matrix(y_pred, y_test):
    VP = 0
    VN = 0
    FP = 0
    FN = 0
    data_final = pandas.DataFrame({'y_pred': y_pred.reshape(len(y_test), ), 'y_test': y_test}).reset_index(drop=True)
    for i in range(len(y_pred)):
        if data_final['y_pred'][i] > 0 and data_final['y_test'][i] == 1:
            VP += 1
        elif data_final['y_pred'][i] > 0 and data_final['y_test'][i] == -1:
            FP += 1
        elif data_final['y_pred'][i] < 0 and data_final['y_test'][i] == 1:
            FN += 1
        elif data_final['y_pred'][i] < 0 and data_final['y_test'][i] == -1:
            VN += 1


    matrix = numpy.reshape((VP, FP, FN, VN), (2, 2))
    col_names = ['Real_True', 'Real_False']
    row_names = ['Pred_True', 'Pred_False']
    conf_m = pandas.DataFrame(matrix, columns = col_names, index = row_names)
    Acc = (VP + VN)/ (VP + VN + FP + FN )
    return conf_m, Acc



