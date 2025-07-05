#!/usr/bin/env python
# coding: utf-8

# # <font color=red> Model de l'evaluació de Espais Vectorials </font>

# ## <font color=green> La prova contindrà preguntes similars a les del model </font>

# <font color=green> Construiu una matriu $B1$ formada pels vectors $\vec{v_1}=(1,2,3,4,5,6,7)$ i $\vec{v_2}=(7,6,5,4,3,2,1)$ com a files. </font>

# In[11]:


v1=vector([1,2,3,4,5,6,7])
v2=vector([7,6,5,4,3,2,1])
matriu=Matrix([v1,v2])
show(matriu)


# <font color=green> Construiu la matriu A formada afegint a la matriu A1 que us donem la matriu $B1$ anterior com a últimes files. No si val a tornar-la a escriure! </font>

# In[12]:


A1=matrix(QQ,5,7,[[0, 1, 2, 3, 4, 5, 6],
 [0, 2, 4, 6, 8, 10, 12],
 [0, 3, 4, 5, 6, 7, 8],
 [0, 6, 5, 4, 3, 2, 1],
 [0, 1, 1, 1, 1, 1, 0]])


# In[13]:


A=A1.transpose().augment(vector(v1)).augment(vector(v2)).transpose()
show(A)


# <font color=green> Construiu una matriu $B$  de mida $4\times 4$ obtinguda prenent les 4 ultimes files i columnes de la matriu A i transposant-la. </font>

# In[14]:


B = A.matrix_from_rows_and_columns([3..6],[3..6]).transpose()
show(B)


# <font color=green> Calculeu la forma esglaonada reduida de $A$ i de $B$. </font>

# In[15]:


AE = A.echelon_form()
show(AE)


# In[16]:


BE = B.echelon_form()
show(BE)


# <font color=green> Calculeu la PAQ-reducció de la matriu B. Cal donar una matriu P i una matriu Q de forma que la matriu $PBQ$ tingui zeros fora de la diagonal i a la diagonal només hi ha 1 o 0. </font>

# In[17]:


New=B.extended_echelon_form()
B_rf=B.extended_echelon_form().matrix_from_columns([0..3])
P=B.extended_echelon_form().matrix_from_columns([4..7])

show("P", P)

show("B", B)

TB_rf=B_rf.transpose();
New2=TB_rf.extended_echelon_form()
transQ=TB_rf.extended_echelon_form().matrix_from_columns([4..7])
PAQred=TB_rf.extended_echelon_form().matrix_from_columns([0..3])
Q=transQ.transpose()

show("Q", Q)

show("PBQ", P*B*Q)


# <font color=green> La següent funció retorna, per a cada matriu A, true o false depenen de si A té una certa forma o no la té. Quina forma? </font>

# In[18]:


def NoSeQueFa(A):
    n=A.nrows()
    m=A.ncols()
    if n!=m:
        return(False)
    for i in range(n):
        for j in range(m):
            if i<j and A[i,j]!=0:
                return(False)
    return(True)


# In[19]:


# Retorna True si la Matriu és triangular inferior, és a dir, que té tot 0s a sobre de la diagonal. A més no importa si els elements de la diagonal són 0 o no. A més, ha de ser quadrada per a que retorni True.


# In[20]:


#Comporvem la funció anterior
M = Matrix(3,3,[1,0,0,2,3,0,2,3,4])
show(M)
print(NoSeQueFa(M))
M = Matrix(3,3,[1,0,0,0,0,0,2,3,4])
show(M)
print(NoSeQueFa(M))
M = Matrix(3,3,[1,0,0-5,0,0,0,2,0,0])
show(M)
print(NoSeQueFa(M))


# <font color=green> Escriu una funció EsIdentitat tal que, donada una matriu quadrada $Q$ retorni true or false depenen de si $Q$ és o no és la matriu identitat. </font>

# In[21]:


def EsIdentitat(Q: Matrix):
    n = Q.nrows()
    m = Q.ncols()
    if n != m:
        return False
    for i in range(n):
        for j in range(m):
            if i==j and Q[i,j] != 1:
                return False
            if i != j and Q[i,j] != 0:
                return False
    return True


# In[22]:


#Comprovem la funció
Q = identity_matrix(5)
show(Q)
print(EsIdentitat(Q))

Q = diagonal_matrix([1,2,3,4])
show(Q)
print(EsIdentitat(Q))

Q = Matrix(3,3,[1,0,0-5,0,0,0,2,0,0])
show(Q)
print(EsIdentitat(Q))


# <font color=green> Definiu el subespai $F$ de $\mathbb{Q}^7$ format per les ultimes 4 files de la matriu $A$. </font>

# In[23]:


show(A)


# In[24]:


F = (QQ^7).span(A.matrix_from_rows([3..6]))
show(F)

basesF = F.basis()
show(basesF)


# <font color=green> Definiu el subespai $G$ de $\mathbb{Q}^7$ format per les ultimes 4 columnes de la matriu $A$. </font>

# In[25]:


G = (QQ^7).span(A.matrix_from_columns([3..6]).transpose())
show(G)

basesG = G.basis()
show(basesG)


# <font color=green> Calculeu una base de la suma $F+G$ i una base de la intersecció $F\cap G$. Cal donar-les com a llistes de vectors. </font>

# In[26]:


suma = (F+G)
show(suma)

suma_bases = suma.basis()
show(suma_bases)


# In[27]:


inter = F.intersection(G).basis()
show(inter)


# In[28]:


#Comprovem que els càlculs estan ben fets:
print(f"Bases de F: {F.dimension()}")
print(f"Bases de G: {G.dimension()}")
print(f"Bases de F+G: {(F+G).dimension()}")
print(f"Bases de la intersecció F∩G: {F.intersection(G).dimension()}")


# <font color=green> Definiu l'aplicació lineal $f$ de $\mathbb{Q}^7$ a $\mathbb{Q}^4$ que té com a matriu la matriu formada per les 4 ultimes columnes de la transposta de $A$. </font>

# In[29]:


f = A.transpose().matrix_from_columns([3..6])
show(f)


# <font color=green> Calculeu bases del subespai nucli i del subespai imatge de $f$. </font>

# In[30]:


kernel = f.kernel() #és el mateix que f.left_kernel()
bases_kernel = kernel.basis()
show(bases_kernel)


# In[31]:


image = f.image()
bases_image = image.basis()
show(bases_image)


# <font color=green> Es $f$ ínjectiva? És exahustiva? </font>

# <font color=red> Poseu aquí la resposta: </font>: 

# <font color=green> Determineu la matriu de l'aplicació lineal $g:\mathbb{Q}^2\to \mathbb{Q}^7$ respecte les bases canòniques de sortida i d'arribada, on $g$ porta el vector $(1,1)$ a la primera fila de $B1$ i el vector $(1,-1)$ a la segona fila de $B1$. </font>

# In[32]:


B1 = matrix(2,2,[1,1,1,-1])
show(B1)


# <font color=green> Determineu la matriu respecte les bases canòniques de sortida i d'arribada de la composició $f\circ g$. </font>

# In[ ]:





# <font color=green> Determineu els valors propis de la matriu $B$ i de la matriu $AA^t$.  </font>

# In[ ]:





# In[ ]:





# <font color=green> Calculeu una matriu invertible $M$ tal que $M^{-1}BM$ sigui diagonal. </font>

# In[ ]:





# In[ ]:





# <font color=green> Calculeu una matriu ortogonal $Q$ amb coeficients reals tal que $Q^tAA^tQ$ sigui diagonal. </font>

# In[ ]:





# In[ ]:





# Calculeu U,V,Sigma de la descomposició en valors singulars per la matriu A. Doneu els valors singulars de A.

# In[33]:


A1 = matrix(CDF, A)
show(A1)
U, S, V = A1.SVD()
show(U)
show(S)
show(V)


# In[34]:


#valors propis
show(A.eigenvalues())


# Escriviu una funció amb Sage que decideixi si una matriu és ortogonal o no.

# In[35]:


print(A.is_orthogonal())

