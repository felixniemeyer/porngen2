PATH=$1
mkdir video-downloads -p
cd video-downloads
yt-dlp --playlist-random --write-info-json "$PATH" -S "height:480"
