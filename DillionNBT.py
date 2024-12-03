# https://minecraft.wiki/w/NBT_format#Binary_format

def getIndexOfKeyInNBT(btyesDataIn: bytes, key: str, startOffset: int = 0):
    keyBytes = key.encode('utf-8')
    firstIndexOfKey = btyesDataIn.find(keyBytes, startOffset)
    
    startOfNBTTag = firstIndexOfKey - 3
    return startOfNBTTag

def decodeTagLongArray(bytesDataIn: bytes, tagStartIndex: int):
    tagID = bytesDataIn[tagStartIndex + 0]
    
    if tagID != 12:
        raise Exception(f"The data provided & the start index ({tagStartIndex}) are not pointing to a TAG_Long_Array")
    
    tagNameLength = int.from_bytes(bytesDataIn[tagStartIndex + 1 : tagStartIndex + 3])
    
    # Get the next 4 bytes after the name ; Int signed, big endian
    arraySizeIntBytes = bytesDataIn[tagStartIndex + tagNameLength + 3 : tagStartIndex + tagNameLength + 7]
    arraySize = int.from_bytes(arraySizeIntBytes, byteorder="big", signed=True)
    
    offsetForArray = tagStartIndex + tagNameLength + 7
    
    longArray = []
    for i in range(0, arraySize):
        nowOffset = offsetForArray + (i * 8) # Long is 8 bytes 
        
        longBytes = bytesDataIn[ nowOffset : nowOffset + 8 ]
        long = int.from_bytes(longBytes, byteorder="big", signed=True)
        
        longArray.append(long)
    
    return longArray


def decodePaletteList(bytesDataIn: bytes, tagStartIndex: int):
    tagID = bytesDataIn[tagStartIndex + 0]
    
    if tagID != 9:
        raise Exception(f"The data provided & the start index ({tagStartIndex}) are not pointing to a TAG_List")
    
    tagNameLength = int.from_bytes(bytesDataIn[tagStartIndex + 1 : tagStartIndex + 3])
    
    # Get the next 4 bytes after the name ; Int signed, big endian
    listSizeIntBytes = bytesDataIn[tagStartIndex + tagNameLength + 3 : tagStartIndex + tagNameLength + 7]
    arraySize = int.from_bytes(listSizeIntBytes, byteorder="little", signed=True)
    
    offsetForArray = tagStartIndex + tagNameLength + 7

    nowOffset = offsetForArray
    lastWasTagEnd = False
    
    for i in range(0, arraySize):
        print(f"Checking data: {nowOffset}, {hex(nowOffset)}")
        
        dataType = bytesDataIn[nowOffset] # Should be 0a (10) -> TAG_Compound
        if dataType != 10: raise Exception(f"Not 0x0a! Is it compound or something I should know? {dataType}")
        # So we're making a json thingie
        
        data = {}
        
        keyDataType = bytesDataIn[nowOffset + 1]
        print(f"Key check thingie ig: {keyDataType}")
        
        if lastWasTagEnd: # TAG_Compound : More JSON
            lengthOfKey = int.from_bytes(bytesDataIn[nowOffset + 2 : nowOffset + 4]) # 4 b/c it's not inclusive :(
            print(f"key - {lengthOfKey}")
        
        if keyDataType == 8:
            lengthOfKey = int.from_bytes(bytesDataIn[nowOffset + 2 : nowOffset + 4]) # 4 b/c it's not inclusive :(
            keyAddressStart = nowOffset + 4
            keyAddressEnd = nowOffset + 4 + lengthOfKey        
            key = (bytesDataIn[keyAddressStart : keyAddressEnd]).decode("utf-8")

            print(f"Key Data {keyDataType} - {lengthOfKey} - '{key}' - {hex(keyAddressEnd)}")
            
            lengthOfValue = int.from_bytes(bytesDataIn[keyAddressEnd : keyAddressEnd + 2]) # 4 b/c it's not inclusive :(
            valueAddressStart = keyAddressEnd + 2
            valueAddressEnd = keyAddressEnd + 2 + lengthOfValue
            value = (bytesDataIn[valueAddressStart : valueAddressEnd]).decode("utf-8")
            
            print(f"Value Data {lengthOfValue} - '{value}' - {hex(valueAddressEnd)}")
            
            data[key] = value
            
        else:
            raise Exception(f"Not 0x08 (TAG_String) nor 0x0a (TAG_Compound)! Key missing or smnth? {keyDataType}")
        
        
        
        isEnd = bytesDataIn[valueAddressEnd]
        print(isEnd)
        if isEnd == 0: lastWasTagEnd = True
        
        print(data)
        
        
        # dataType (1), keyDataType (1), getKeyLength (2), lengthOfKey (whatever value it is), lengthOfValue (2),
        # lengthOfValue (whatever value), checkEndByte (1)
        nowOffset += 1 + 1 + 2 + lengthOfKey + 2 + lengthOfValue + 1


