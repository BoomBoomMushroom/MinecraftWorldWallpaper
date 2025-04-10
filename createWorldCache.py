import chunkReader
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import json

regionFilesFolder = f"./world/region"
regionCoords = (0, 0)
x, y = regionCoords

regionFileName = f"r.{x}.{y}"
regionFileNameWithExtension = f"{regionFileName}.mca"
allChunks = chunkReader.read_region_file(f"{regionFilesFolder}/{regionFileNameWithExtension}", forceNoCache=False)

def subChunkGenerator(allChunks):
    for chunk in allChunks:
        for subchunk in chunk:
            yield subchunk

subchunks = subChunkGenerator(allChunks)

# Create the cache folder
regionCacheFolderPath = f"{regionFilesFolder}/{regionFileName}"
try:
    os.mkdir(regionCacheFolderPath)
except: pass

for subchunk in subchunks:
    subchunkPosition = subchunk[2]
    fileName = f"{subchunkPosition[0]},{subchunkPosition[1]},{subchunkPosition[2]}.json"
    with open(f"{regionCacheFolderPath}/{fileName}", "w") as f:
        f.write(json.dumps(subchunk))

