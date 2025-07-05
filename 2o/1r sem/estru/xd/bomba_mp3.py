import os.path
import vlc

path = os.getcwd()

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.lower().endswith(tuple(['.mp3'])):
            file = os.path.join(root, filename)
            print("found:  " + filename)
            player = vlc.MediaPlayer(file)
            player.play()