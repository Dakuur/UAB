import os.path
import vlc
import eyed3

path = os.getcwd()

print("I still think I am the greatest. - Kanye West")

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.lower().endswith(tuple(['.mp3'])):
            file = os.path.join(root, filename)
            if eyed3.load(file).tag.artist == "Kanye West":
                print("found: " + filename)
                player = vlc.MediaPlayer(file)
                player.play()