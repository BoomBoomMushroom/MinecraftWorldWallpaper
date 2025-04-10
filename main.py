import chunkReader
import chunkRenderer
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cProfile
import pstats

worldCoordsToRender = [
    
]

chunkCoordsToRender = [
    
]

regionFilesToRender = [
    
]

def addBoundBoxToChunkCoordToRender(pointA: tuple[int, int], pointsB: tuple[int, int]):
    bottomLeft = (min(pointA[0], pointsB[0]), min(pointA[1], pointsB[1]))
    topRight = (max(pointA[0], pointsB[0]), max(pointA[1], pointsB[1]))
    
    for x in range(bottomLeft[0], topRight[0]):
        for z in range(bottomLeft[1], topRight[1]):
            chunkCoordsToRender.append((x, z))

addBoundBoxToChunkCoordToRender((8, 24), (20, 27))


def getLargestChunkIndexInChunkCoordsToRender():
    biggestIndex = -1
    for chunkCoord in chunkCoordsToRender:
        chunkIndex = chunkCoord[0] + chunkCoord[1] * 64
        biggestIndex = max(biggestIndex, chunkIndex)
    
    return biggestIndex

def getSmallestChunkIndexInChunkCoordsToRender():
    biggestIndex = 64 + 64 * 64
    
    for chunkCoord in chunkCoordsToRender:
        chunkIndex = chunkCoord[0] + chunkCoord[1] * 64
        biggestIndex = min(biggestIndex, chunkIndex)
    
    return biggestIndex

# Convert world coordinates into chunk coordinates
for worldCoord in worldCoordsToRender:
    chunkCoordsToRender.append((
        math.floor(worldCoord[0] / 16),
        math.floor(worldCoord[1] / 16)
    ))

for chunkCoord in chunkCoordsToRender:
    x = math.floor(chunkCoord[0]) >> 5
    z = math.floor(chunkCoord[1]) >> 5
    regionFileName = f"r.{x}.{z}.mca"
    regionFilesToRender.append(regionFileName)
    #print(regionFileName)

def renderChunks():
    chunkRenderer.renderFrame()

def loadChunks(chunks):
    #"""
    for chunk in chunks:
        for subChunk in chunk:
            
            # only get subchunks above a certain level
            if subChunk[2][1] < 50 / 16: continue
            
            chunkRenderer.loadSubChunkFromJson(subChunk)
    #"""
    
    """
    allSubChunks = [subChunk for chunk in chunks for subChunk in chunk]
    with ThreadPoolExecutor() as executor:
        executor.map(chunkRenderer.loadSubChunkFromJson, allSubChunks)
    del allSubChunks
    """ 
    
def generateAndApplyTextureArray():
    #print(chunkRenderer.uniqueTextures, len(chunkRenderer.uniqueTextures))
    textureArray = chunkRenderer.generateTextureArray(chunkRenderer.ctx, chunkRenderer.uniqueTextures, 16, 16)
    textureArray.use() # apply it
    return textureArray

def cullFacesWithNaiveMeshing():
    #chunkRenderer.removeUnseenFaces(chunkRenderer.blocks)
    for chunkKey in chunkRenderer.chunks.keys():
        chunkRenderer.removeUnseenFaces(chunkRenderer.chunks[chunkKey])

def doGreedyMesher():
    return chunkRenderer.distributeGreedyMeshAlgorithm()

# TODO: I need to cut down on memory usage!
def main():
    chunkRenderer.blocks = []
    chunksUnfiltered = chunkReader.read_region_file(f"./world/region/{regionFileName}", forceNoCache=False)
    chunks = []
    #chunks = chunksUnfiltered
    
    #"""
    # Filter the chunks w/ only the correct position
    #offset = 3 * 32
    for chunk in chunksUnfiltered:
        #if offset > 0:
        #    offset -= 1
        #    continue
        
        if len(chunks) >= 32 * 8: break
        
        chunks.append(chunk)
    #"""
    
    if len(chunks) == 0: return
    del chunksUnfiltered

    print("starting to load chunks")
    loadChunks(chunks)
    print("Done loading chunks")

    del chunks

    # Generate textures
    
    print(f"Starting to generate and apply texture array")
    textureArray = generateAndApplyTextureArray()
    print(f"Done Texture Array! It has {textureArray.layers} layers")


    # Cull faces (Naive Meshing)
    print("starting to cull unseen faces")
    cullFacesWithNaiveMeshing()
    print("done removing unseen faces")
    print(f"Number of blocks remaining: {len(chunkRenderer.blocks)}")


    # TODO: Idea to speed up the greedy mesher
    # Greedy Mesh per-chunk, which should also potentially solve the problem of meshing too big of a face (cough cough, bottom bedrock layer)
    # Also greedy meshing per-chunk should also (hopefully) allow us Thread it even more. Thread each type of face for each chunk at the same time

    # Greedy Meshing
    print("Starting the greedy mesh algorithm")
    facesLeftFromGreedyMeshing = doGreedyMesher()
    print(f"Number of faces from greedy meshing algorithm: {facesLeftFromGreedyMeshing}")

    #firstBlock = chunkRenderer.blocks[0]
    firstBlock = chunkRenderer.chunks[list(chunkRenderer.chunks.keys())[0]][0]
    chunkRenderer.camera.position.x = firstBlock["x"] + 5
    chunkRenderer.camera.position.y = firstBlock["y"] + 1
    chunkRenderer.camera.position.z = firstBlock["z"] + 5

    print(chunkRenderer.camera.position)
    #"""
    
    print("Cleaning up memory...")
    del chunkRenderer.chunks
    del chunkRenderer.blockMap
    del chunkRenderer.blockMapKeys
    del chunkRenderer.uniqueTextures
    del chunkRenderer.textureCache
    del chunkRenderer.vertex_shader
    del chunkRenderer.fragment_shader
    
    del chunkRenderer.specialBlocksToFullFaces
    del chunkRenderer.blockTextureOverride
    del chunkRenderer.blockTextureOverrideKeys
    del chunkRenderer.blocksToSkip
    del chunkRenderer.faceNameToFaceIndex
    del chunkRenderer.neighborOffsets
    del chunkRenderer.defaultTexturePath
    print("Done cleaning up memory")

    didDelAllInstanceData = False

    print("Starting to render!")
    running = True
    while running:
        running = chunkRenderer.renderFrame()
        
        if didDelAllInstanceData == False:
            del chunkRenderer.allInstanceData
            print("Deleted allInstanceData")
            didDelAllInstanceData = True


with cProfile.Profile() as pr:
    main()


stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats() # Print The Stats
stats.dump_stats("./cprofile.prof") # Saves the data in a file, can me used to see the data visually
