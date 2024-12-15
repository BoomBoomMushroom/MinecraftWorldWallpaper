import struct
import json
# py -m pip install bitarray
from bitarray import bitarray

import subprocess

def getCorrectIndices(blockIndices, numberOfBlocksInPalette, discardExtraBits=True):
    bitsNeededForIndex = max(
        4,
        numberOfBlocksInPalette.bit_length()
    ) # 4 is the minimum number of bits
    
    #print(struct.pack("q", 0xFA))  # b'\xfa\x00\x00\x00\x00\x00\x00\x00'
    #print(struct.pack(">q", 0xFA)) # b'\x00\x00\x00\x00\x00\x00\x00\xfa'
    
    def bitArrayToIndices(indexBitArray, bitsNeededForIndex):
        blockIndicesReal = []
        while len(indexBitArray) >= bitsNeededForIndex:
            getIndexBits = indexBitArray[-bitsNeededForIndex:]
            indexBitArray = indexBitArray[:-bitsNeededForIndex]
            
            getIndex = int(getIndexBits.to01(), 2)
            blockIndicesReal.append(getIndex)
        
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

def read_region_file(file_path):
    """Read and parse a single region (.mca) file."""
    # https://minecraft.wiki/w/Region_file_format#Structure
    print(f"Reading chunk file: {file_path}")
    
    regionFileJsonResult = subprocess.run(["nbt-to-json-rust.exe", f"--input={file_path}", "--type=REGION"], stdout=subprocess.PIPE)
    regionFileJsonString = regionFileJsonResult.stdout.decode("utf-8")
    regionFileJson = json.loads(regionFileJsonString)
    
    chunks = []
    
    for chunk in regionFileJson.values():
        chunks.append([])
        
        chunkDataVersion = chunk["DataVersion"]
        isOneDotSixteenOrHigher = chunkDataVersion >= 2566
        
        positionChunkCords = [ chunk["xPos"], chunk["yPos"], chunk["zPos"] ]
        positionWorldCords = [ chunk["xPos"]*16, chunk["yPos"]*16, chunk["zPos"]*16 ]
        
        print(positionChunkCords, positionWorldCords)
        
        for subchunk in chunk["sections"]:
            if "block_states" not in subchunk: continue
            
            blockStates = subchunk["block_states"]
            blockPalette = blockStates["palette"]
            blockIndicesReal = [0] * 4096 # Fill this up with the first object, because if the palette only has 1 object, then this would error because all blocks in the 16x16x16 are the same block
            
            numberOfBlocksInPalette = len(blockPalette)
            if numberOfBlocksInPalette > 1:
                blockIndicesReal = getCorrectIndices(blockStates["data"], numberOfBlocksInPalette-1, isOneDotSixteenOrHigher)
            
            chunks[-1].append([
                blockIndicesReal,
                blockPalette,
                [ positionWorldCords[0], subchunk["Y"] * 16, positionWorldCords[2] ],
            ])
    
    return chunks

if __name__ == "__main__":
    worldPath = "D:/DillionPC/Coding/Github/MinecraftWorldWallpaper/world"
    
    data = read_region_file(worldPath + "/region/r.0.0.mca")
    print(data)