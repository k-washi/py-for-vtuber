import numpy as np

class Positionlkf():
  def __init__(self, init_x, init_y):
    
    self.T = 1.
    #state space model x = A*x + B*u + w, w ~ N(0, Q)
    self.A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    self.B = np.array([[0,0, self.T, 0], [0, 0, 0, self.T], [0, 0, 0, 0], [0, 0, 0, 0]])
    self.Q = np.array([[50, 0, 0, 0],[0, 50, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]]) #移動誤差

    #observe model y = C * x + v, v ~ N(0, R)
    self.C = np.array([[1,0, 0, 0], [0,1, 0, 0]])
    self.R = np.array([[200, 0],[0, 200]]) #計測誤差

    self.x = np.array([[init_x], [init_y], [0], [0]])

    self.mu = np.array([[0], [0], [0], [0]])
    self.Sigma = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    self.mu_ = np.array([[0], [0], [0], [0]])
    self.Sigma_ = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) 

    self.e_w = init_x
    self.e_h = init_y
    self.uw = 0
    self.uh = 0

  def estimate(self):
    #estiarrayion
    
    self.mu_ = np.dot(self.A, self.x) + np.dot(self.B, np.array([[0], [0], [self.uw], [self.uh]])) #u速度の内界情報としてtrackerの移動量を与える #現在の推定値
    print("mu_ {0}, {1}, {2}".format(self.mu_, self.uw, self.x))
    self.Sigma_ = self.Q + np.dot(np.dot(self.A, self.Sigma), self.A.T) #現在の誤差行列
    return np.array([self.mu_[0][0], self.mu_[1][0]]), self.Sigma_

  def update(self, w, h):
    #update
    #鼻先の位置を与える
    yi = np.array([[w], [h]]) - np.dot(self.C, self.mu_) #観測残差
    #print("yi {0}".format(yi))
    S = np.dot(np.dot(self.C, self.Sigma_), self.C.T) + self.R #観測残差の共分散
    K = np.dot(np.dot(self.Sigma_, self.C.T), np.linalg.inv(S))  #カルマンゲイン
    self.x = self.mu_ + np.dot(K, yi)#更新された現在の推定値
    self.Sigma = self.Sigma_ - np.dot(np.dot(K, self.C), self.Sigma_)#更新された現在の誤差行列
    #print("x {0}".format(self.x))

    return np.array([self.x[0][0], self.x[1][0]]), self.Sigma
  
  def calcVelocity(self, w, h):
    self.uw = w - self.e_w
    self.uh = h - self.e_h
    self.e_w = w
    self.e_h = h

    #return uw, uh
  
