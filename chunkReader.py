import struct
import json
# py -m pip install bitarray
from bitarray import bitarray
from concurrent.futures import ThreadPoolExecutor
import os

import subprocess

def getCorrectIndices(blockIndices, numberOfBlocksInPalette, discardExtraBits=True):
    bitsNeededForIndex = max(
        4,
        numberOfBlocksInPalette.bit_length()
    ) # 4 is the minimum number of bits
    
    def bitArrayToIndices(indexBitArray, bitsNeededForIndex):
        bitArrayLength = len(indexBitArray)
        blockIndicesReal = [
            int(indexBitArray[start:start + bitsNeededForIndex].to01(), 2)
            for start in range(bitArrayLength - bitsNeededForIndex, -1, -bitsNeededForIndex)
        ]
        return blockIndicesReal

    
    blockIndicesReal = []
    allsBitsArray: bitarray = bitarray()
    
    for blockStateIndex in blockIndices:
        indexBytes = struct.pack(">q", blockStateIndex)
        indexBitArray: bitarray = bitarray()
        indexBitArray.frombytes(indexBytes)
        
        if discardExtraBits == False:
            allsBitsArray.extend(indexBitArray)
            continue
        
        indices = bitArrayToIndices(indexBitArray, bitsNeededForIndex)
        blockIndicesReal.extend(indices)
        
    
    if discardExtraBits == False:
        blockIndicesReal = bitArrayToIndices(allsBitsArray, bitsNeededForIndex)
    
    return blockIndicesReal

def calculate_subchunk(subchunk, positionWorldCords, isOneDotSixteenOrHigher):
    if "block_states" not in subchunk: return None
    
    blockStates = subchunk["block_states"]
    blockPalette = blockStates["palette"]
    blockIndicesReal = [0] * 4096 # Fill this up with the first object, because if the palette only has 1 object, then this would error because all blocks in the 16x16x16 are the same block
    
    numberOfBlocksInPalette = len(blockPalette)
    if numberOfBlocksInPalette > 1:
        blockIndicesReal = getCorrectIndices(blockStates["data"], numberOfBlocksInPalette-1, isOneDotSixteenOrHigher)
    
    return [
        blockIndicesReal,
        blockPalette,
        [ positionWorldCords[0], subchunk["Y"] * 16, positionWorldCords[2] ],
    ]

def get_subchunks_from_chunk(chunk):
    subchunks = []
    
    chunkDataVersion = chunk["DataVersion"]
    isOneDotSixteenOrHigher = chunkDataVersion >= 2566
    
    positionChunkCords = [ chunk["xPos"], chunk["yPos"], chunk["zPos"] ]
    positionWorldCords = [ chunk["xPos"]*16, chunk["yPos"]*16, chunk["zPos"]*16 ]
    
    print(positionChunkCords, positionWorldCords)
    
    for subchunk in chunk["sections"]:
        section = calculate_subchunk(subchunk, positionWorldCords, isOneDotSixteenOrHigher)
        if section == None: continue
        
        subchunks.append(section)
    
    return subchunks

def read_region_file(file_path, forceNoCache=False):
    """Read and parse a single region (.mca) file."""
    # https://minecraft.wiki/w/Region_file_format#Structure
    
    chunks = []
    
    decodedFileName = file_path.split(".")
    decodedFileName.pop()
    decodedFileName = ".".join(decodedFileName) + ".json"
    
    if os.path.exists(decodedFileName) and forceNoCache == False:
        print("A pre-decoded file exists... going to use it")
        with open(decodedFileName, "r") as f:
            chunks = json.load(f)
            return chunks
    
    print(f"Reading chunk file: {file_path}")
    
    regionFileJsonResult = subprocess.run(["nbt-to-json-rust.exe", f"--input={file_path}", "--type=REGION"], stdout=subprocess.PIPE)
    regionFileJsonString = regionFileJsonResult.stdout.decode("utf-8")
    regionFileJson = json.loads(regionFileJsonString)
    regionFileChunks = regionFileJson.values()
    
    with ThreadPoolExecutor() as executor:
        chunks = list(executor.map(get_subchunks_from_chunk, regionFileChunks))
    
    # Cache this file
    with open(decodedFileName, "w") as f:
        f.write( json.dumps(chunks) )
    
    """
    for chunk in regionFileChunks:
        subchunks = load_chunk(chunk)
        chunks.append(subchunks)
    """
    
    return chunks

if __name__ == "__main__":
    worldPath = "D:/DillionPC/Coding/Github/MinecraftWorldWallpaper/world"
    
    data = read_region_file(worldPath + "/region/r.0.0.mca")
    print(data)