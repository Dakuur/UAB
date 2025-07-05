def join(arr1: list, arr2: list) -> list:
    if arr1[-1] < arr2[0]:
        return arr1 + arr2
    else:
        return arr2 + arr1

def mergesort(arr: list) -> list:

    n = len(arr)
    print(f"len: {n}")

    if n <= 1: # 1 ELEMENT
        return arr
    elif n == 2: # SORT 2
        if arr[0] <= arr[1]:
            return arr
        else:
            return [arr[1], arr[0]]
            
    mid = n//2
    print(f"mid: {mid}")
    arr1 = arr[:mid]
    arr2 = arr[mid:]

    res1 = mergesort(arr1)
    res2 = mergesort(arr2)

    return join(res1, res2)

l = [1,3,5,2]

print(mergesort(l))