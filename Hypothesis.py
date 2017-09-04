import numpy as np
import copy
import threading

def loss(y_pred, y):
    return (y_pred - y) ** 2 / 2.

def deloss(y_pred, y):
    return (y_pred - y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def desigmoid(y):
    return y * (1. - y)

class Hypothesis:
    def __init__ (self, W, b):
        """
        W: weight for each layer (dict)
        b: bias unit for each layer (matrix)
        """
        self.W = W
        self.b = b
        self.G = {}

    def forward(self, x):
        W = self.W; depth = len(W)
        a = {}
        ### FORWARD ###
        a[0] = x	# shape=[1, x_dim]
        b = np.ones((len(a[0]), len(self.b.reshape(-1)))) * self.b.reshape(-1)
        for t in range(depth):
            a[t] = np.c_[a[t], b[:,t]]
            a[t+1] = sigmoid(a[t].dot(W[t]))
        a[depth] = a[depth-1].dot(W[depth-1])
        return a

    def forward_w(self, w, x):
        depth = len(w)
        a = {}
        ### FORWARD ###
        a[0] = x.reshape(1, -1)	# shape=[1, x_dim]
        b = np.ones((len(a[0]), len(self.b.reshape(-1)))) * self.b.reshape(-1)
        for t in range(depth):
            a[t] = np.c_[a[t], b[:,t]]
            a[t+1] = sigmoid(a[t].dot(w[t]))
        a[depth] = a[depth-1].dot(w[depth-1])
        return a

    def backpropagte_gradient(self, x, y, lr, lam):
        """
               d             d z(t+1)       d a(t+1)      d Loss
            ------ Loss() = ---------- * ( ----------- * --------- )
            d W(t)            d W(t)        d z(t+1)      d a(t+1)
                                |                      |_____________ backward pass
                                |_______________ forward pass
    
                                    ||	
                 sigmoid			||	
            z(t) -------> a(t) --->	|| ---> a(t) -------> z(t+1)
                                    ||			  W(t)
                                    ||	
                                    ||	

        - forward pass
            : because a(t)*W(t) = z(t+1), the forward pass = a(t)

        - backward pass
            : the backward pass = desigmoid(a(t)) * d_Loss/d_a(t),
              discuss in two parts, 

            [ OUTPUT LAYER ]
                backward_pass  = d_z(T)/d_z(T) * d_Loss/d_z(T)
                               = 1 * deloss(z(T), y)

                   => delta(T) = deloss(z(T), y)

            [ NOT OUTPUT LAYER ]
                backward_pass = d_a(t)/d_z(t) * d_Loss/d_a(t)
                              = desigmoid(a(t)) * ( W(t) * desigmoid(a(t+1)) * delta(t+1) )

                      d a(t)  
                (1)  --------  = desigmoid(a(t))
                      d z(t)  

                       d Loss    d z(t+1)   d a(t+1)    d Loss
                (2)  --------- = -------- * -------- * ---------
                       d a(t)     d a(t)    d z(t+1)    d a(t+1)

                   => delta(t) = W(t) * desigmoid(a(t+1)) * delta(t+1)
            
        """
        W = self.W; depth = len(W)
        delta = {}; W_new = {}
        a = self.forward(x)
        de_a = {}
        ### BACKWARD ###
        #------------------------------------#
        # a[t]			: (1, K[t] + 1)
        # delta[t]		: (1, K[t] + 1)
        # delta[t+1]	: (1, K[t+1] + 1)
        # desigmoid[t+1]: (1, K[t+1] + 1)
        # W[t]			: (K[t] + 1, K[t+1])
        #------------------------------------#
        ### delta(T) ###
        delta[depth] = deloss(a[depth], y)
        delta[depth] = np.c_[delta[depth], np.ones((len(delta[depth]), 1))]
        ### desigmoid(a(t)), and desigmoid(a(T)) = 1 ###
        for t in range(depth):
            de_a[t] = desigmoid(a[t]) #shape=[1, K[t]+1]
        de_a[depth] = np.ones((len(a[depth]), 2))
        ### delta(t) where t = 0 ~ T-1 ###
        for t in reversed(range(depth)):
            delta[t] = (de_a[t+1][:,:-1] * delta[t+1][:,:-1]).dot(W[t].T)
            # i.e. => delta[t] = (W[t].dot( (desigmoid(a[t+1][:,:-1]) * delta[t+1][:,:-1]).T )).T

        ### GRADIENT ###
        for t in range(depth):
            u = np.kron(a[t], (de_a[t+1][:,:-1] * delta[t+1][:,:-1])).reshape(W[t].shape)
            W_new[t] = W[t] - lr * u - lr * lam * W[t]
            # print("new W[{}] : {}".format(t, W_new[t]))
        return W_new

    def gradient(self, x, label, lr, lam):
        self.G = {}; gradient = self.G
        th = {}
        h = 0.00001
        W = self.W; depth = len(W)
        x = x.reshape(1, -1)
        th_id = 0
        for t in range(len(W)):
            gradient[t] = np.zeros(W[t].shape)
            for i in range(W[t].shape[0]):
                for j in range(W[t].shape[1]):
                    th[th_id] = threading.Thread(target=self.th_gradient, args=(x, label, t, i, j))
                    th[th_id].start()
                    th_id += 1
                    # W_new = copy.deepcopy(W)
                    # W_new[t][i][j] = W[t][i][j] + h
                    # above = loss(self.forward_w(W_new, x)[depth], label)
                    # W_new[t][i][j] = W[t][i][j] - h
                    # below = loss(self.forward_w(W_new, x)[depth], label)
                    # # Compute the numeric gradient.
                    # sample = (above - below) / (2 * h)
                    # gradient[t][i][j] = sample

        for i in range(th_id):
            th[i].join()
        W_new = {}
        for t in range(depth):
            W_new[t] = W[t] - lr * gradient[t] - lr * lam * W[t]
        return W_new

    def th_gradient(self, x, label, t, i, j):
        gradient = self.G
        h = 0.00001
        W = self.W; depth = len(W)
        W_new = copy.deepcopy(W)
        W_new[t][i][j] = W[t][i][j] + h
        above = loss(self.forward_w(W_new, x)[depth], label)
        W_new[t][i][j] = W[t][i][j] - h
        below = loss(self.forward_w(W_new, x)[depth], label)
        # Compute the numeric gradient.
        sample = (above - below) / (2 * h)
        gradient[t][i][j] = sample

    def train(self, data, label, lr=0.0002, lam=0.5):
        """
        - x: input data (vector)
        - W: all weight for each layer (dict)
          |-- len(W): the count of layers (i.e. depth)
          |-- W[t].shape[0]: input layer size + 1
          |-- W[t].shape[1]: output layer size
        - lr: learning rate
        - lam: regularization parameter
        [RETURN]
          W_new = W + lr * (u) - lr * lam * W
        """
        W_new = self.backpropagte_gradient(data.reshape(1,-1), label, lr, lam)
        # W_new = self.gradient(data.reshape(1,-1), label, lr, lam)
        self.W = W_new

        a = self.forward(data.reshape(1,-1))
        # print("==> output: {}, y: {}".format(a[len(self.W)], label))
        # print("==> Loss: {}".format(loss(a[len(self.W)], label)))
        return W_new

    def error(self, data, label):
        depth = len(self.W)
        a = self.forward(data)
        h_label = a[depth]
        return np.mean(loss(label, h_label))

    def update(self, W, b):
        self.W = W
        self.b = b

