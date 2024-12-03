import get_chunk
import chunkRenderer
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cProfile
import pstats

def renderChunks():
    chunkRenderer.renderFrame()

def getChunksFromRegionFile(fileName):
    chunks = get_chunk.read_region_file(fileName, 128, 0, -1, 7)
    
    #with open("chunks.json", "w") as f:
    #    f.write(json.dumps(chunks, indent=4))
    
    return chunks

def loadChunks(chunks):
    #"""
    for chunk in chunks:
        for subChunk in chunk:
            chunkRenderer.loadSubChunkFromJson(subChunk)
    #"""
    
    """
    with ThreadPoolExecutor() as executor:
        allSubChunks = [subChunk for chunk in chunks for subChunk in chunk]
        executor.map(chunkRenderer.loadSubChunkFromJson, allSubChunks)
    """ 
    
def generateAndApplyTextureArray():
    #print(chunkRenderer.uniqueTextures, len(chunkRenderer.uniqueTextures))
    textureArray = chunkRenderer.generateTextureArray(chunkRenderer.ctx, chunkRenderer.uniqueTextures, 16, 16)
    textureArray.use() # apply it
    return textureArray

def cullFacesWithNaiveMeshing():
    chunkRenderer.removeUnseenFaces(chunkRenderer.blocks)

def doGreedyMesher():
    return chunkRenderer.distributeGreedyMeshAlgorithm()

def main():
    chunkRenderer.blocks = []
    chunks = getChunksFromRegionFile("./world/region/r.0.0.mca")

    print("starting to load chunks")
    loadChunks(chunks)
    print("Done loading chunks")

    # Generate textures
    
    print(f"Starting to generate and apply texture array")
    textureArray = generateAndApplyTextureArray()
    print(f"Done Texture Array! It has {textureArray.layers} layers")


    # Cull faces (Naive Meshing)
    print("starting to cull unseen faces")
    cullFacesWithNaiveMeshing()
    print("done removing unseen faces")
    print(len(chunkRenderer.blocks))


    # TODO: Idea to speed up the greedy mesher
    # Greedy Mesh per-chunk, which should also potentially solve the problem of meshing too big of a face (cough cough, bottom bedrock layer)
    # Also greedy meshing per-chunk should also (hopefully) allow us Thread it even more. Thread each type of face for each chunk at the same time

    # Greedy Meshing
    print("Starting the greedy mesh algorithm")
    facesLeftFromGreedyMeshing = doGreedyMesher()
    print(f"Number of faces from greedy meshing algorithm: {facesLeftFromGreedyMeshing}")


    firstBlock = chunkRenderer.blocks[0]
    chunkRenderer.camera.position.x = firstBlock.x + 5
    chunkRenderer.camera.position.y = firstBlock.y + 1
    chunkRenderer.camera.position.z = firstBlock.z + 5

    print(chunkRenderer.camera.position)
    #"""

    running = True
    while running:
        running = chunkRenderer.renderFrame()


with cProfile.Profile() as pr:
    main()


stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats() # Print The Stats
stats.dump_stats("./cprofile.prof") # Saves the data in a file, can me used to see the data visually
