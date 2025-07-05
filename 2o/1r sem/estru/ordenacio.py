import time
from random import randrange

def ordenacioIntercanvi(v):
    for passades in range(1,len(v)):
      for i in range(len(v)-passades):
  	    if v[i] > v[i+1]:              
              v[i],v[i+1]=v[i+1],v[i]   

def cerca (v,x):
    trobat = False
    i=0
    while (i<len(v)) and (not trobat):
        trobat = (v[i] == x)
        i+=1			        
    return trobat

def intercanvi(v,i,j):
    aux=v[i]
    v[i]=v[j]
    v[j]=aux
    #v[i],v[j]=v[j],v[i]  

def mulMatVect(m,v):
    if len(m)!=len(v):
        raise Exception('ERROR: Matriu i Vector tenen mides incompatibles')
    vRes=[0]*len(m)
    for fila in range(len(m)):
      vRes[fila]=0
      for col in range(len(v)):
          vRes[fila]+=m[fila][col]*v[col]        
    return vRes
          
if __name__=='__main__':
    
    MAXVAL=10000
    v=[v*randrange(MAXVAL) for v in range(MAXVAL)]
    
   
    print("COMENCEM INTERCANVI")
    start_time = time.time()    # 1
    intercanvi(v,0,MAXVAL-1)
    end_time = time.time()      # 2
    run_time = end_time - start_time    # 3
    print(f"Intercanvi fet en {run_time:.4f} segs")
    
    
    print("COMENCEM ORDENACIO")
    start_time = time.time()    # 1
    ordenacioIntercanvi(v)
    end_time = time.time()       # 2
    run_time = end_time - start_time    # 3
    print(f"Ordenacio feta en {run_time:.4f} segs")
    
    print("COMENCEM CERCA ULTIM")
    start_time = time.time()    # 1
    trobat = cerca(v,v[MAXVAL-1])
    end_time = time.time()     # 2
    run_time = end_time - start_time    # 3
    if (trobat):
        print("Trobat")
    else:
        print("NO TROBAT")
    print(f"Cerca feta en {run_time:.4f} segs")
    
    print("COMENCEM CERCA PRIMER")
    start_time = time.time()     # 1
    trobat = cerca(v,v[0])
    end_time = time.time()      # 2
    run_time = end_time - start_time    # 3
    if (trobat):
        print("Trobat")
    else:
        print("NO TROBAT")
    print(f"Cerca feta en {run_time:.4f} segs")
    
    print("MULTIPLICA")
    start_time = time.time()     # 1
    m=[]
    
    for i in range(MAXVAL):
        l=[x*randrange(MAXVAL) for x in range(MAXVAL)]
        m.append(l)