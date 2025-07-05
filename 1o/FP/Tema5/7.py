def centenes(n):
    if int(n) < 100:
        cent = 0
    else:
        cent = str(n[-3])
    return cent

num = input()

print("Les centenes del nombre",num,"són",centenes(num))