import random

class MapBase:

    class Item:
        #Parella de clau,valor que és element de la taula hash.
        __slots__ = ["_key" ,"_value"]

        def __init__ (self, k, v):
            self._key = k
            self._value = v

        def __eq__ (self, other):
            return self._key == other._key
        
        def __ne__ (self, other):
            return not (self == other) # oposat de eq
        
        def __lt__ (self, other):
            return self._key < other._key # defineix ordre segons clau
        
        def __str__(self):
            return "(" + str(self._key)+":"+str(self._value) + ")"
        
        def __repr__(self):
            return "(" + str(self._key)+":"+str(self._value) + ")"
        
        @property
        def key(self):
            return self._key
        
        @key.setter
        def key(self, k):
            self._key = k

        @property
        def value(self):
            return self._value
        
        @value.setter
        def value(self, v):
            self._value = v

    """Definicio class Hash Simple sense funció
    hash nomes accedeix per clau"""

    def __init__ (self):
        """Crea una taula hash buida."""
        self.__table = [ ] # utilitzem una llista

    def __getitem__ (self, k):
        """Retorna self[k], KeyError si no trobat"""
        for item in self.__table:
            if k == item.key:
                return item.value
        raise KeyError(' Key Error: '+ repr(k))
    
    def __setitem__ (self, k, v):
        """assigna self[k]=v
        i si ja tenia valor sobreescriu"""
        for item in self.__table:
            if k == item.key:
                item.value=v
                return self.__table.append(self.Item(k,v))
    
    def __delitem__ (self, k):
        """esborra self[k], KeyError si no trobat"""
        for j in range(len(self.__table)):
            if k == self.__table[j].key: # Found a match
                self.__table.pop(j) # remove item
                return # and quit
        raise KeyError(' Key Error: '+ repr(k))
    
    def __len__ (self):
        """Return number of items in the map."""
        return len(self.__table)
    
    def __iter__ (self):
        """Generate iteration of the map s keys."""
        for item in self.__table:
            yield item
            
    def __str__(self):
        return str(self.__table)
    
    def __repr__(self):
        return str(self.__table)

class Hash(MapBase):
   
    def __init__ (self,cap=11, p=109345121):
        """Crea una taula hash buida."""
        self._table = cap * [ None ]
        self._n = 0 # number of entries in the map
        self._prime = p # prime for MAD compression
        self._scale = 1 + random.randrange(p-1) # scale from 1 to p-1 for MAD
        self._shift = random.randrange(p) # llista Items

    def _hash_function(self, k):
        return (hash(k)* self._scale + self._shift) % self._prime % len(self._table)
    
    def __iter__ (self):
        for el in self._table:
            if el is not None: # a nonempty slot
                yield el

    
    def __contains__ (self, k):
        try:
            self[k] # access via getitem (ignore result)
            return True
        except KeyError:
            return False
        
    def setdefault(self, k, d):
        try:
            return self[k] # if getitem succeeds, return value
        except KeyError: # otherwise:
            self[k] = d # set default value with setitem
            return d    
        
    def __getitem__ (self, k):
        """Retorna valor associat a clau k (raise KeyError si no el troba)."""
        j = self._hash_function(k)
        if self._table[j] is None:
            raise KeyError('Key Error: ' + repr(k))
        if (self._table[j]._key==k):
            return self._table[j]._value
        else:
            raise KeyError('Key Error: COLLISIO' + repr(k) + "COLLISIO AMB " + repr(self._table[j]._key))
            
    def __setitem__ (self, k, v):
        """Assign value v to key k, overwriting existing value if present."""
        pos = self._hash_function(k)
        if (self._table[pos] is None):
            self._table[pos]=self.Item( k, v) 
            self._n+=1
        elif (self._table[pos]._key==k):
            self._table[pos]._value=v            
        else:
           raise KeyError('Key Error: COLLISIO' + repr(k) + "COLLISIO AMB " + repr(self._table[pos]._key))
        if self._n > 0.75 * len(self._table): # keep load factor <= 0.5
            self._resize(2 *len(self._table) - 1)

    def __delitem__ (self, k):
        """Remove item associated with key k (raise KeyError if not found)."""
        j = self._hash_function(k)
        if (self._table[j] is None):
            raise KeyError('Key No Existeix: ' + repr(k) )
        if not (self._table[j]._key==k):
            raise KeyError('Key Error: COLLISIO' + repr(k) + "COLLISIO AMB " + repr(self._table[j]._key))
        
        del self._table[j]
        self._n-=1

    def __len__ (self):
        """Return number of items in the map."""
        return self._n
    
    def __str__(self):
        return str(self._table)
    
    def __repr__(self):
        return str(self._table)
    
    """def _resize(self,c):
        old = [i for i in self] # use iteration to record existing items
        self._table = c * [None] # then reset table to desired capacity
        self._n = 0 # n recomputed during subsequent adds
        for item in old:
            try:
                self[item._key]=item._value          
            except KeyError: # otherwise:
                #print("Comment :=>> COLLISIO " , e[0])
                print("Comment :=>> COLLISIO")
        old=None"""
    
a = Hash()
a["houda"] = 2
print(a)
a[4234] = "gfdkghlfd"
print(a)
a.__delitem__("houda")
print(a)
a = dict()
a["si"] = 0
a.__delitem__("si")
print(a)