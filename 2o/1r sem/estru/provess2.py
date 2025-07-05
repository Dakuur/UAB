# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:45:58 2019

@author: gemma
"""

import sys

l1=[300]*5
l2=[300 for x in range(5)]
l3=[300,300,300,300,300]

print("MIDA l1: ", sys.getsizeof(l1))
print("MIDA l2: ", sys.getsizeof(l2))
print("MIDA l3: ",sys.getsizeof(l3))

l4=[1000,500,700,800,900]
l5=l4[1:4]
print(l5)
l5[1]=523
print("l4: despres l5[1]=23 ", l4)
print("l5: despres l5[1]=23 ", l5)

a=[1,2]
b=[3,4]
c=[5,6]
l1=[a,b]
l2=l1
l3=list(l1)
print("l1: ",l1)
print("l2: ",l2)
print("l3: ",l3)

l1.append(c)

print("l1: despres l1.append(c) : ",l1)
print("l2: despres l1.append(c) :  ",l2)
print("l3: despres l1.append(c) :  ",l3)

l2[0][1]=55
print("l1: despres l2[0][1]=55 : ",l1)
print("l2: despres l2[0][1]=55 :  ",l2)
print("l3: despres l2[0][1]=55 :  ",l3)
print("a l2[0][1]=55: " , a)