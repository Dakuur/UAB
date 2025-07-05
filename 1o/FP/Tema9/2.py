def is_anagram(str1, str2):
    return (sorted(str1) == sorted(str2))

str1 = input("Introdueix la primera cadena: ")
str2 = input("Introdueix la segunda cadena: ")

if is_anagram(str1, str2):
    print("És un anagrama")
else:
    print("No és un anagrama")