class A:
    def __init__(self) -> None:
        self.s = 2
        
        self.list = [self.s]
    
    def show(self):
        print(self.list)
    
    
a = A()

while 1:
    a.s+=1 
    a.show()