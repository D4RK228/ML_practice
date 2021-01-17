```python
class MyClass(object):
    def __init__(self, value, parent1, parent2, operation):
        self.value = value
        self.parent1 = parent1
        self.parent2 = parent2
        self.operation = operation
        self.grad = 0
        self.grad1 = 1
        self.grad2 = 1
            
        
    def ComputeGrads(self):
        if self.operation == "mul":
            self.grad1 = self.parent2.value
            self.grad2 = self.parent1.value
        if self.operation == "sub":
            self.grad1 = 1
            self.grad2 = -1
        if self.operation == "add":
            self.grad1 = 1
            self.grad2 = 1
        if self.operation == "truediv":
            self.grad1 = 1/self.parent2.value
            self.grad2 = -1*(self.parent1.value/(self.parent2.value**2))
            
        if self.parent1 != None and self.parent2 != None:
            self.parent2.grad += self.grad2 * self.grad
            self.parent1.grad += self.grad1 * self.grad 
            self.parent1.ComputeGrads()
            self.parent2.ComputeGrads()  
            


    def __mul__(self, other):
        value = self.value * other.value
        return MyClass(value, self, other, "mul")

    def __add__(self, other):
        value = self.value + other.value
        return MyClass(value, self, other, "add")

    def __sub__(self, other):
        value = self.value - other.value
        return MyClass(value, self, other, "sub")

    def __truediv__(self, other):
        value = self.value / other.value
        return MyClass(value, self, other, "truediv")
    
    
class Dense(object):
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.weights = [[MyClass(random(), None, None, None) for j in range(self.size_in)] for i in range(self.size_out)]
        
    def __call__(self, vector):
        arr = []
        for i in self.weights:
            s = 0
            for j in range(self.size_in):
                s += i[j] * vector[j]
            arr.append(s)
        return arr


def dfs(x):
    if x.parent1:
        dfs(x.parent1)
    if x.parent2:
        dfs(x.parent2)
    if x not in dfs_list:
        dfs_list.append(x)

def ComputeTopGrads(dfs_list):
    dfs_list.reverse()
    for self in dfs_list:
        if self.operation == "mul":
            self.grad1 = self.parent2.value
            self.grad2 = self.parent1.value
        if self.operation == "sub":
            self.grad1 = 1
            self.grad2 = -1
        if self.operation == "add":
            self.grad1 = 1
            self.grad2 = 1
        if self.operation == "truediv":
            self.grad1 = 1/self.parent2.value
            self.grad2 = -1*(self.parent1.value/(self.parent2.value**2))
        if self.parent1 != None and self.parent2 != None:
            self.parent2.grad += self.grad2 * self.grad
            self.parent1.grad += self.grad1 * self.grad 
            
            

a = MyClass(15, None, None, None)
b = MyClass(4, None, None, None)
c = a - b
f = c*c

dfs_list = []
dfs(f)
f.grad = 1
ComputeTopGrads(dfs_list)
f.grad

```
