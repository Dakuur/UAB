class David:

    def __init__(self, valors: dict) -> None:
        self._valors = valors

    def __iter__(self):
        for i in self._valors:
            yield i

x = {
    "gat": "miau",
    "gos": "guau",
    "ocell": "pip"
}

a = David(x)

"""for i in a:
    print(i)

for i in x:
    print(type(i))"""

from GrafHash import GrafHash

hash_dict = GrafHash()

hash_dict[1] = "hello"
hash_dict[2] = "bye"
hash_dict[234] = "the"

del hash_dict[2]

print(hash_dict)

from GrafHash import GrafHash

dic_hash = GrafHash(cap=200)

dic_hash[1] = "hello"
dic_hash["David"] = "houda"
dic_hash[3] = "adeu"
dic_hash[325] = "si"
dic_hash["pol"] = "depende"
dic_hash[66] = 757

del dic_hash[1]

print(dic_hash)