import numpy as np 

class MinDistance: 
    def __init__(self): 
        self.points, self.weights = self.initialize()

    def initialize(self): 
        x, y = 1, 1
        weights = np.ones(10)
        points = np.zeros((10, 2))

        for i in range(10):
             
            dx = np.random.normal()
            dy = np.random.normal()
            if i == 9: 
                points[i][:] = np.array([x, y]) * 10
                weights[i] = 10
            else: 
                points[i][:] = np.array([x - dx, y - dy])

        return points, weights

    def stochastic_gradient(self, index, x): 
        return self.weights[index] * (x - self.points[index]) 

    def get_param_dim(self):
        return self.points.shape  

    def evaluate(self, x):
        fn = 0 
        for i, point in enumerate(self.points): 
            fn += (self.weights[i] * np.linalg.norm(x - point) ** 2)
        
        return fn 

    def get_param(self):
        return self.points, self.weights

    def analytical_solution(self): 
        den = 1. / np.sum(self.weights)
        num = np.zeros(2) 
        for i, point in enumerate(self.points): 
            num += (self.weights[i] * point)

        return num * den 
         

    def get_lipschitz_proba(self): 
        return self.weights / np.sum(self.weights)

if __name__ == '__main__': 
    obj = MinDistance()
    print(obj.analytical_solution())