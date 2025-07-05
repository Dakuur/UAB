from pytube import YouTube
YouTube('https://www.youtube.com/watch?v=EMlM6QTzJo0').streams.first().download()
yt = YouTube('https://www.youtube.com/watch?v=EMlM6QTzJo0')
yt.streams