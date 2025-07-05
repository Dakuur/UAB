def funcassert(n):
    assert n>0, "Ha de ser positiu"
    assert n%2==0, "Ha de ser múltiple de 2"

def funcexcept(n):
    if n>0:
        raise ValueError("Ha de ser positiu")
    if n%2==0:
        raise ValueError("Ha de ser múltiple de 2")

funcexcept(-8)
funcexcept(10)