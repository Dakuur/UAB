def checker_v1(num: int):
    if num < 0:
        return
    elif num > 10:
        return
    else:
        return num + 10
    
def checker_v2(num: int):
    if num > 0:
        if num < 10:
            return num + 10
    return

def checker_v3(num: int):
    if 0 <= num <= 10:
        return num + 10
    return

def checker_v4(num: int):
    return num + 10 if 0 <= num <= 10 else None