import pyautogui
import time

time.sleep(3)

"""
file = open("common.txt", "r")

for line in file:
    pyautogui.write(line[:-1])
    pyautogui.press('enter')

file.close()

for i in range(0, 101):
    pyautogui.write(str(i))
    pyautogui.press('enter')
"""

for i in "abcdefghijklmnopqrstuwxyz":
    for x in "abcdefghijklmnopqrstuwxyz":
        pyautogui.write(str(i) + str(x))
        pyautogui.press('enter')