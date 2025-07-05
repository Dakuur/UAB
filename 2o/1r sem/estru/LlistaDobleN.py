class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class LlistaDobleN:

    def __init__(self, llista = None):
        self.head = None
        self.tail = None
        if llista != None:
            for item in llista:
                self.append_final(item)

    def __eq__(self, other):
        if isinstance(other, LlistaDobleN):
            #LlistaDobleEnllaçada
            current_self = self.head
            current_other = other.head

            while current_self is not None and current_other is not None:
                if current_self.data != current_other.data:
                    return False
                current_self = current_self.next
                current_other = current_other.next
            return current_self is None and current_other is None

        elif isinstance(other, list):
            #llista normal
            current = self.head
            for item in other:
                if current is None or current.data != item:
                    return False
                current = current.next
            return current is None
        return False
    
    def get_size(self):
        size = 0
        current = self.head
        while current is not None:
            size += 1
            current = current.next
        return size

    def append_final(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
    
    def append_principi(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

    def append(self, data):
        self.append_final(data)

    def inserirPosicio(self, val, pos):
        new_node = Node(val)

        if pos == 0:
            self.append_principi(val)

        elif pos == -1:
            self.append_final(val)

        else:
            current = self.head
            index = 0
            while current is not None and index < pos:
                current = current.next
                index += 1

            if current is None:
                raise ValueError("Posició no vàlida")

            new_node.next = current
            new_node.prev = current.prev
            if current.prev is not None:
                current.prev.next = new_node
            current.prev = new_node

    def inserirPosicioList(self, pos, l2):
        if pos == -1:
            pos = self.get_size()

        if pos < 0 or pos > self.get_size():
            raise ValueError("Posició no vàlida")

        if pos == 0:
            for item in reversed(l2):
                self.append_principi(item)
        else:
            current = self.head
            index = 0
            while index < pos - 1:
                current = current.next
                index += 1

            for item in reversed(l2):
                new_node = Node(item)
                new_node.next = current.next
                new_node.prev = current
                if current.next is not None:
                    current.next.prev = new_node
                current.next = new_node

    def delete(self, valor):
        current = self.head

        while current is not None:
            if current.data == valor:
                if current.prev is not None:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next is not None:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                return

            current = current.next

        raise ValueError(f"El valor {valor} no se encuentra en la lista")

    def reverse(self):
        current = self.head
        while current is not None:
            current.prev, current.next = current.next, current.prev
            current = current.prev
        self.head, self.tail = self.tail, self.head

    def __getitem__(self, index):
        if not (-self.get_size() <= index < self.get_size()):
            raise IndexError("Índice fuera de rango")

        if index >= 0:
            current = self.head
            for i in range(index):
                current = current.next
        else:
            current = self.tail
            for i in range(abs(index) - 1):
                current = current.prev
        return current.data

    def __str__(self):
        values = []
        current = self.head
        while current is not None:
            values.append(str(current.data))
            current = current.next
        return '[' + ', '.join(values) + ']'

l1 = [1,1,3,10,5,6,69]

l2 = LlistaDobleN(l1)
print(l2)

posicioEliminar = [ 1 , 2, 3, 0, -1, 0, -1 ]
i = 0
l2.delete(l2[-1])

l2.append(100)