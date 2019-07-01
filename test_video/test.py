import os
from PIL import Image
from ffmpy3 import FFmpeg
import subprocess
import ffmpeg

in_jpgDatasetPath= 'det_1.avi'
outname = 'det_111.mp4'

getmp3='ffmpeg' +' -i '+in_jpgDatasetPath+' -c h264 '+outname
print(getmp3)
returnget= subprocess.call(getmp3,shell=True)