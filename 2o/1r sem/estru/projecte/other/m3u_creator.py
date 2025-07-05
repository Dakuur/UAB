import os
import eyed3
import cfg
import random

def create_m3u_file(folder_path, output_path):
    # Create a list to store the file entries
    i = 0

    with open(output_path, "w") as m3u_file:
        m3u_file.write("#EXTM3U\n")

        # Iterate through the files in the specified folder
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(".mp3"):
                    mp3_path = os.path.join(root, filename)

                    # Use EyeD3 to get the artist and title
                    audiofile = eyed3.load(mp3_path)
                    try:
                        artist = audiofile.tag.artist
                        title = audiofile.tag.title
                    except:
                        artist = "Unknown Artist"
                        title = "Unknown Title"

                    # Format the entry and add it to the list
                    canon = cfg.get_canonical_pathfile(mp3_path)
                    entry = f"#EXTINF:-1, {artist} - {title}\n{canon}"
                    try:
                        m3u_file.write(f"{entry}\n")
                        i += 1
                    except:
                        pass
    print(f"Added {i} songs")

def create_m3u_file_random(folder_path, output_path, probability=0.5):
    # Create a list to store the file entries
    entries = []

    # Iterate through the files in the specified folder
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".mp3") and random.random() < probability:
                mp3_path = os.path.join(root, filename)

                # Use EyeD3 to get the artist and title
                audiofile = eyed3.load(mp3_path)
                try:
                    artist = audiofile.tag.artist
                    title = audiofile.tag.title
                except:
                    artist = "Unknown Artist"
                    title = "Unknown Title"

                # Format the entry and add it to the list
                canon = cfg.get_canonical_pathfile(mp3_path)
                entry = f"#EXTINF:-1, {artist} - {title}\n{canon}"
                entries.append(entry)

    # Shuffle the list of entries
    random.shuffle(entries)

    # Open the file after leaving the 'with' block
    with open(output_path, "w") as m3u_file:
        m3u_file.write("#EXTM3U\n")

        # Write the shuffled entries to the m3u file
        for entry in entries:
            try:
                m3u_file.write(f"{entry}\n")
            except:
                pass

    print(f"Added {len(entries)} songs")

# Example usage:
folder_path = cfg.get_root()
output_path = "10.m3u"
#create_m3u_file(folder_path, output_path)
create_m3u_file_random(folder_path, output_path, probability=0.5)