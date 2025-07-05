import copy

class Pila:
    def __init__ (self,l=None):
        if l==None:
            self._data = [ ]
        else:
            self._data = list(l)
            
    def __len__ (self):
        return len(self._data)
         
    def is_empty(self):
        return len(self._data) == 0
     
    def push(self, e):
        self._data.append(e)
    
    def top(self):
        assert not self.is_empty(), "La pila està buida"
        return self._data[-1]
     
    def pop(self):
        assert not self.is_empty(), "La pila està buida"
        return self._data.pop()
     
    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
       return self._data.__repr__()

if __name__ == "__main__":
    p=Pila([])
    print(p.top())
   
    