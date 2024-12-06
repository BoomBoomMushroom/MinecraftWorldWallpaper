# https://minecraft.wiki/w/NBT_format#Binary_format

def getIndexOfKeyInNBT(bytesDataIn: bytes, key: str, startOffset: int = 0):
    keyBytes = key.encode('utf-8')
    firstIndexOfKey = bytesDataIn.find(keyBytes, startOffset)
    
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

def readTagString(bytesDataIn: bytes, tagStartIndex: int):
    tagID = bytesDataIn[tagStartIndex + 0]
    
    if tagID != 8:
        raise Exception(f"The data provided & the start index ({tagStartIndex}) are not pointing to a TAG_String")
    
    stringName = readNextFewBytesAsString(bytesDataIn, tagStartIndex + 1)
    
    return stringName

def readNextFewBytesAsString(bytesDataIn: bytes, startIndex: int):
    stringLength = int.from_bytes(bytesDataIn[startIndex + 0 : startIndex + 2])
    stringNameBytes = bytesDataIn[startIndex + 2 : startIndex + 2 + stringLength]
    #print(stringLength, stringNameBytes)
    stringName = stringNameBytes.decode("utf8")
    return stringName

def readNextTag(bytesDataIn: bytes, tagStartIndex: int):
    tagID = bytesDataIn[tagStartIndex + 0]
    
    return readNextTagOfKnownType(bytesDataIn, tagStartIndex, tagID)
    
def readNextTagOfKnownType(bytesDataIn: bytes, tagStartIndex: int, tagID: int):
    match tagID:
        case 0: pass
        case 8:
            return readNextFewBytesAsString(bytesDataIn, tagStartIndex)
        case _: raise Exception(f"Not any TagID I know of! Tag ID: {tagID} ~ Address: {tagStartIndex} / {hex(tagStartIndex)}")
    

def decodePaletteList(bytesDataIn: bytes, tagStartIndex: int):
    tagID = bytesDataIn[tagStartIndex + 0]
    
    if tagID != 9:
        raise Exception(f"The data provided & the start index ({tagStartIndex}) are not pointing to a TAG_List")
    
    tagNameLength = int.from_bytes(bytesDataIn[tagStartIndex + 1 : tagStartIndex + 3])
    
    # Get the next 4 bytes after the name ; Int signed, big endian
    listSizeIntBytes = bytesDataIn[tagStartIndex + tagNameLength + 3 : tagStartIndex + tagNameLength + 7]
    arraySize = int.from_bytes(listSizeIntBytes, byteorder="little", signed=True)
    
    palette = []
    
    offsetForArray = tagStartIndex + tagNameLength + 7
    for i in range(arraySize):
        #print(f"Array Index for palette {i} : Size of palette {arraySize}")
        
        indexOfPropertiesStart = getIndexOfKeyInNBT(bytesDataIn, "Properties", offsetForArray)
        indexOfNameStart = getIndexOfKeyInNBT(bytesDataIn, "Name", offsetForArray)
        
        #print(palette)
        #print(hex(indexOfNameStart), hex(indexOfPropertiesStart))
        
        if indexOfPropertiesStart < 0:
            # Make it bigger than the name address to let it choose the name
            indexOfPropertiesStart = indexOfNameStart + 1
            #print("HEYYY WHY ARE WE LESS?!?!")

            if indexOfNameStart < 0: break
        
        blockThingData = {}
        
        def getNameTagAndValue():
            nameKey = readTagString(bytesDataIn, indexOfNameStart)
            #print(f"Name out: {nameKey}")
            
            nameValue = readNextFewBytesAsString(bytesDataIn, indexOfNameStart + 3 + len(nameKey))
            #print(f"Name value out: {nameValue}")
            
            return nameKey, nameValue # Set offsetForArray
        
        
        
        # Properties is typically first, so if it isn't the first index, then we can (probably) assume that this element has no properties
        if indexOfPropertiesStart > indexOfNameStart:
            nameKeyShouldBeProperties, nameValue = getNameTagAndValue()
            blockThingData[nameKeyShouldBeProperties] = nameValue
            
            offsetForArray = indexOfNameStart + 4
        else:
            #print(hex(indexOfPropertiesStart))
            nameKeyShouldBeProperties = readNextFewBytesAsString(bytesDataIn, indexOfPropertiesStart+1)
            
            properties = {}
            
            propertyType = bytesDataIn[indexOfPropertiesStart + 3 + len(nameKeyShouldBeProperties)]
            #print(propertyType, hex(propertyType))
            
            propertyNameAddress = indexOfPropertiesStart + 3 + len(nameKeyShouldBeProperties) + 1
            propertyName = readNextFewBytesAsString(bytesDataIn, propertyNameAddress)
            #print(f"Property Name: {propertyName}")
            
            propertyValueAddress = propertyNameAddress + 2 + len(propertyName)
            propertyValue = readNextTagOfKnownType(bytesDataIn, propertyValueAddress, propertyType)
            
            #print(f"Property value: {propertyValue} : address {propertyValueAddress} / {hex(propertyValueAddress)}")
            
            properties[propertyName] = propertyValue
            
            nameNameKey, nameValue = getNameTagAndValue()
            
            blockThingData[nameNameKey] = nameValue
            blockThingData[nameKeyShouldBeProperties] = properties
            
            offsetForArray = indexOfNameStart + 4
        
        #print(blockThingData)
        palette.append(blockThingData)

    return palette
    


