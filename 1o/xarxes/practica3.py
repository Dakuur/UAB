#-------------------6--------------------

# Grup: c7
# NIU: 1665191 - Alumne 1: Adria Muro Gomez
# NIU: 1666540 - Alumne 2: David Morillo Massague
FILE = "quotes.json"

import json
import time
import socket

servers = ["djxmmx.net", "djxmmx.net", "djxmmx.net"]
result = []

def get_quote(servername):
    port = 17
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((servername, port))
    quote = sock.recv(1024).decode().strip()
    return quote

i = 0

while len(result) < 31:
    quote = get_quote(servers[i])
    if quote not in result:
        result.append(quote)
    else:
        pass
    time.sleep(2)
    if i < 2:
        i += 1
    else:
        i = 0

with open(FILE, "w") as file:
    json.dump(result, file)



#-------------------7--------------------
# Grup: c7
# NIU: 1665191 - Alumne 1: Adria Muro Gomez
# NIU: 1666540 - Alumne 2: David Morillo Massague

import socket
import random
import json

HOST = ''  	 
PORT = 8307
FILE = "quotes.json"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

with open(FILE, "r") as quotes_file:
	quotes = json.load(quotes_file)

while True:
	conn, addr = s.accept()
	print('Connected by', addr)
    
	random_quote = random.choice(quotes)
	quote_data = json.dumps(random_quote).encode('utf-8')
    
	conn.send(quote_data)
	conn.close()
