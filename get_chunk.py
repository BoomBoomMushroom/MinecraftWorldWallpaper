import os
import zlib
import math
import struct
import json
# py -m pip install "nbtlib==1.12.1"
import nbtlib
# py -m pip install bitarray
from bitarray import bitarray

import DillionNBT


def byteArrayToIntArrayOfNBits(byteArrayData, numBits):
    """Turns a bytearray into an array of unsigned ints, the amount of bits each int has is defined by n_bits

    Args:
        byteArrayData (_type_): The input bytearray to turn into ints
        numBits (_type_): The number of bits each int will have
        
    Returns:
        _type_: A list of ints, each with N Bits
    """
    result = []
    bit_string = ''.join(f'{byte:08b}' for byte in byteArrayData)  # Convert byte array to a bit string
    totalAmountOfBits = len(bit_string)
    
    for i in range(0, totalAmountOfBits, numBits):
        # Extract `n_bits` bits at a time
        bits = bit_string[i:i+numBits]
        
        # If the last group of bits is less than `n_bits`, we ignore it
        if len(bits) == numBits:
            # Convert the extracted bits to an unsigned integer
            result.append(int(bits, 2))
        else:
            #result.append(int(bits, 2))
            print(f"[byteArrayToIntArrayOfNBits] WE ARE IGNORING THE LAST BYTE OF SIZE {len(bits)}")
    
    return result

def unpackLongArray_SkipTooLittle(byteArrayData, numBits):
    """Turns a bytearray into an array of unsigned ints, the amount of bits each int has is defined by n_bits

    Args:
        byteArrayData (_type_): The input bytearray to turn into ints
        numBits (_type_): The number of bits each int will have
        
    Returns:
        _type_: A list of ints, each with N Bits
    """
    result = []
    
    longSizeInBits = 64
    longSizeInBytes = int(longSizeInBits/8)
    # len * 8 to get the # of bits, not # of bytes
    numOfLongs = round( (len(byteArrayData)*8) / longSizeInBits )
    
    #print(numOfLongs)
    
    for i in range(0, numOfLongs):
        longAddressStart = i*longSizeInBytes
        longAddressEnd = longAddressStart + longSizeInBytes
        longBytes = byteArrayData[longAddressStart:longAddressEnd]
        
        longBitString = ''.join(f'{byte:08b}' for byte in longBytes)
        
        #print(longBitString)
        
        out = []
        for j in range(0, len(longBitString), numBits):
            bitsLeft = len(longBitString)-j
            if numBits > bitsLeft: continue
            
            newIntBits = longBitString[j:j+numBits]
            newInt = int(newIntBits, 2)
            
            #print(longBitString)
            #out.append(newIntBits)
            out.append(newInt)
            
        #print(out, len(out), longSizeInBits/numBits)
        result.extend(out)
    
    return result

def getCorrectIndices(blockIndices, numberOfBlocksInPalette, discardExtraBits=True):
    bitsNeededForIndex = max(
        4,
        math.ceil( math.log2(numberOfBlocksInPalette) )
    ) # 4 is the minimum number of bits
    
    #print(struct.pack("q", 0xFA))  # b'\xfa\x00\x00\x00\x00\x00\x00\x00'
    #print(struct.pack(">q", 0xFA)) # b'\x00\x00\x00\x00\x00\x00\x00\xfa'
    
    blockIndicesReal = []
    
    for blockStateIndex in blockIndices:
        indexBytes = struct.pack(">q", blockStateIndex)
        indexBitArray: bitarray = bitarray()
        indexBitArray.frombytes(indexBytes)
        
        while len(indexBitArray) >= bitsNeededForIndex:
            getIndexBits = indexBitArray[-bitsNeededForIndex:]
            indexBitArray = indexBitArray[:-bitsNeededForIndex]
            
            getIndex = int(getIndexBits.to01(), 2)
            blockIndicesReal.append(getIndex)
            
            #if numberOfBlocksInPalette == 18:
            #    print(len(indexBitArray), bitsNeededForIndex, getIndex, numberOfBlocksInPalette)
    
    return blockIndicesReal


def getChunksHeaderData(regionFile):
    allChunksHeaderData = []
    
    regionFile.seek(0x0000, 0)
    for i in range(0x0FFF): # Will stop on 0x0FFF
        # Chunk Location
        # First 3 Bytes (Big Endian) are offsets in 4 KiB sectors from the start of the file
        # Last (4th) byte is the length of the chunk (also in 4 KiB sector, rounded up) ! Always less than 1 KiB in size
        # If a chunk isn't present both (all?!) fields are zero
        offsetBytes = regionFile.read(3)
        sizeByte = regionFile.read(1)
        
        offset = int.from_bytes(offsetBytes, byteorder="big", signed=False)
        size = int.from_bytes(sizeByte)
        
        chunkMemoryLocation = offset * 4096
        
        #print(offsetBytes, sizeByte)
        #print(offset, size)
        
        allChunksHeaderData.append([chunkMemoryLocation, size])
    return allChunksHeaderData

def getChunkTimestamps(regionFile):
    regionFile.seek(0x1000, 0)
    timestamps = []
    for i in range(0x0FFF): # Will stop on 0x1FFF
        timestampBytes = regionFile.read(4) # 4 byte big endian, last time the chunk was modified (in epoch seconds)
        timestamp = int.from_bytes(timestampBytes, byteorder="big", signed=False)
        
        timestamps.append(timestamp)
    
    return timestamps

def decompressChunkData(compressedChunkData, compressionType):
    uncompressedChunkData = bytearray()
    
    # Compression Schemes: https://minecraft.wiki/w/Region_file_format#Structure:~:text=There%20are%20currently,information%20needed%5D
    if compressionType == 1:
        # TO_DO: Implement Compression Type: GZip (RFC1952) (unused in practice)
        # Probably don't because it is unused in practice
        raise NotImplementedError("Compression Type Not implemented! GZip (RFC1952) (unused in practice)")
    elif compressionType == 2:
        try:
            uncompressedChunkData = zlib.decompress(compressedChunkData)
        except zlib.error as e:
            print(f"Failed to uncompress using Zib: {e}")
            
    elif compressionType == 3:
        uncompressedChunkData = compressedChunkData
    elif compressionType == 4:
        # TO_DO: Implement Compression Type: LZ4 (since 24w04a, enabled in server.properties)
        # Gonna comment this out so that color doesn't bother me 
        raise NotImplementedError("Compression Type Not implemented! LZ4 (since 24w04a, enabled in server.properties)")
    elif compressionType == 127:
        # TO_DO: Implement Compression Type: Custom compression algorithm (since 24w05a, for third-party servers)
        # Also probably not since I have no reason, plus I don't know where to start. So if this is an issue, I'll figure it out
        # Check the Compression Schemes link above the first "if" statement if we need to implement this
        raise NotImplementedError("Compression Type Not implemented! Custom compression algorithm (since 24w05a, for third-party servers)")
    
    #       >  If the value of compression scheme increases by 128, the compressed data is saved in a file called c.x.z.mcc, where x and z are the chunk's coordinates, instead of the usual position.
    # So like ummm, I will NOT be figuring this rn, since I haven't experienced this
    return uncompressedChunkData

def chunkDataToNBT(chunkData):
    # I can't load the bytearray directly into nbtlib, so I need to make it into a file first ;-; 
    with open("chunk.nbt", "wb") as chunkTemp:
        chunkTemp.write(chunkData)
    
    with open("chunk.nbt", "rb") as chunkTemp:
        nbtChunkData = nbtlib.File.from_fileobj(chunkTemp)
        
        try:
            nbtChunkData = nbtChunkData[""]
        except: pass # I guess not all of the time ¯\_(ツ)_/¯
        
        # for some reason the json is structured like this; so lets access that first bit to get the real data
        # {
        #   "": {REAL DATA HERE}
        # }
        return nbtChunkData

def read_region_file(file_path, chunksToRead=1, startChunkIndex=0, subChunksToRead=1, startSubChunkIndex=0):
    """Read and parse a single region (.mca) file."""
    # https://minecraft.wiki/w/Region_file_format#Structure
    print(f"Reading chunk file: {file_path}")
    
    chunks = []
    
    with open(file_path, 'rb') as mcaFile:
        # Getting chunk info (location in memory and size)
        allChunksHeaderData = getChunksHeaderData(mcaFile) # 1024 Entries; 4 bytes each; Each index will be an array, 0 is location in memory, 1 is how much memory for that chunk
        
        # Get chunk modification timestamps
        #timestamps = getChunkTimestamps(mcaFile)
        
        # Reading Chunk Data (Payload)
        chunkNumber = 0
        
        for chunkNumber in range(startChunkIndex, len(allChunksHeaderData)):
        #for chunkHeaderData in allChunksHeaderData:
            if chunksToRead > 0 and chunkNumber >= chunksToRead+startChunkIndex:
                break
        
            chunks.append([]) # This empty list is our subchunks
        
            chunkHeaderData = allChunksHeaderData[chunkNumber]
            chunkNumber += 1
            print(f"Chunk {chunkNumber} / {len(allChunksHeaderData)}")
            
            locationInMemory = chunkHeaderData[0]
            #chunkDataSize = chunkHeaderData[1]
            
            mcaFile.seek(locationInMemory, 0)
            
            # Minecraft always pads the last chunk's data to be a multiple-of-4096B in length
            #   (so that the entire file has a size that is a multiple of 4KiB). Minecraft does not accept files in which
            #   the last chunk is not padded. Note that this padding is not included in the length field.
            
            # Rest of the data is compressed data (length-1 bytes)
            lengthInBytes = int.from_bytes( mcaFile.read(4), byteorder="big", signed=True )
            compressionType = int.from_bytes( mcaFile.read(1), signed=True ) # Probably false since the highest compression scheme is 127, which is max for a signed 8-bit number
            
            compressedChunkData = mcaFile.read(lengthInBytes - 1)
            uncompressedChunkData = decompressChunkData(compressedChunkData, compressionType)
            
            dataIndices = []
            paletteIndices = []
            dataNextSearchOffset = 0
            
            while True:
                blockStateIndex = DillionNBT.getIndexOfKeyInNBT(uncompressedChunkData, "block_states", dataNextSearchOffset)
                if blockStateIndex < 0: break
                #print(blockStateIndex)
                
                longIndex = DillionNBT.getIndexOfKeyInNBT(uncompressedChunkData, "data", blockStateIndex)
                if longIndex < 0: break
                #print(longArr)
                
                paletteIndex = DillionNBT.getIndexOfKeyInNBT(uncompressedChunkData, "palette", longIndex)
                print(paletteIndex, hex(paletteIndex))
                #paletteIndex = paletteIndices
                DillionNBT.decodePaletteList(uncompressedChunkData, paletteIndex)
                
                dataIndices.append(longIndex)
                dataNextSearchOffset = longIndex + 8
                
                break
            
            print(dataIndices, len(dataIndices))
            
            """
            for subChunkNumber in range(startSubChunkIndex, len(dataIndices)):
                #blockPalette = blockStates["palette"]
                
                dataPositionInMemory = subChunkNumber
                blockIndicesPacked = DillionNBT.decodeTagLongArray(dataPositionInMemory)
                blockIndicesReal = [0] * 4096
                
                if len(blockIndicesPacked) > 1:
                    blockIndicesReal = getCorrectIndices(blockIndicesPacked, numberOfBlocksInPalette-1, True)
            """
            
            #print(uncompressedChunkData)
            #with open("./nbt.lw", "wb") as f:
            #    f.write(uncompressedChunkData)
            
            #out = DillionNBT.get_specific_index(uncompressedChunkData, "biomes")
            #print(out)
            
            """
            nbtChunkData = chunkDataToNBT(uncompressedChunkData)
            
            #nbtChunkJson = nbtChunkData.snbt(indent=4)
            #with open("Chunk.json", "w") as chunkJson:
            #    chunkJson.write(nbtChunkJson)
            
            if nbtChunkData == {}: continue
            
            # Cool, now all data is in this structure/format:
            # https://minecraft.wiki/w/Chunk_format

            # https://minecraft.fandom.com/wiki/Data_version
            chunkDataVersion = nbtChunkData["DataVersion"]
            isOneDotSixteenOrHigher = chunkDataVersion >= 2566

            print(f"Chunk data version: {chunkDataVersion} | Is past 1.16? {isOneDotSixteenOrHigher}")

            positionChunkCords = [ nbtChunkData["xPos"], nbtChunkData["yPos"], nbtChunkData["zPos"] ]
            positionWorldCords = [ nbtChunkData["xPos"]*16, nbtChunkData["yPos"]*16, nbtChunkData["zPos"]*16 ]
            #inhabitedTime = nbtChunkData["InhabitedTime"]
            #lastUpdate = nbtChunkData["LastUpdate"]
            chunkSections = nbtChunkData["sections"] # 16x16x16 area.
            
            print(positionChunkCords, positionWorldCords)
            
            #print(blockSections, len(blockSections))
            for subChunkNumber in range(startSubChunkIndex, len(chunkSections)):
            #for chunkSection in chunkSections:
                if subChunksToRead > 0 and subChunkNumber >= startSubChunkIndex+subChunksToRead:
                    break
                
                print(f"SubChunk {subChunkNumber} / {len(chunkSections)}")
                
                chunkSection = chunkSections[subChunkNumber]
                
                blockStates = chunkSection["block_states"]
                blockPalette = blockStates["palette"]
                blockIndicesReal = [0] * 4096 # Fill this up with the first object, because if the palette only has 1 object, then this would error because all blocks in the 16x16x16 are the same block
                
                print(len(blockStates["data"]))
                
                numberOfBlocksInPalette = len(blockPalette)
                if numberOfBlocksInPalette > 1:
                    blockIndicesReal = getCorrectIndices(blockStates["data"], numberOfBlocksInPalette-1, isOneDotSixteenOrHigher)
                
                
                chunks[-1].append([
                    blockIndicesReal,
                    blockPalette,
                    [ positionWorldCords[0], chunkSection["Y"] * 16, positionWorldCords[2] ],
                ])
                
                #print(position, chunkSection["Y"])
                #break
            """
            
            #break
        
        #with open("OutSubchunk.json", 'w') as subChunkJsonFile:
        #    subChunkJsonFile.write( json.dumps(subChunks) )
        
        #try: os.remove("chunk.nbt")
        #except: pass
        
        return chunks

def get_all_chunks(world_path):
    """Extract all chunks from a Minecraft world."""
    region_dir = os.path.join(world_path, "region")
    chunks = []
    for file_name in os.listdir(region_dir):
        if file_name != "r.0.0.mca": continue
        
        if file_name.endswith(".mca"):
            file_path = os.path.join(region_dir, file_name)
            
            fileChunks = read_region_file(file_path)
            chunks.extend(fileChunks)
    
    return chunks

if __name__ == "__main__":
    worldPath = "D:/DillionPC/Coding/Github/MinecraftWorldWallpaper/world"
    #chunks = get_all_chunks(worldPath)
    
    data = read_region_file("./world/region/r.0.0.mca", 1, 0, 1, 0)
    print(data)