def area_quadrat(costat):
     if costat>0:
          area=costat*costat
          return 0,area
     else:
          return 1,None
def area_rectangle(base,altura):
     if (base>0)and(altura>0):
          area=base*altura
          return 0,area
     else:
          return 1,None
def area_triangle(base,altura):
     if (base>0)and(altura>0):
          area=base*altura/2
          return 0,area
     else:
          return 1,None
