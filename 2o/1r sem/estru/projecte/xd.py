import cfg
import os

root = cfg.get_root()
current_directory = os.getcwd()

files = os.listdir(root)

"""for file in files:
    print(file)"""

c = str(root + "/blues.m3u")
with open(c, 'r', encoding="latin-1") as file:
    entire_program = file.read()
    print(entire_program)
    
raise