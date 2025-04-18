from __future__ import annotations
import pygame
import moderngl
import numpy as np
from pygame.locals import *
from PIL import Image
from glm import mat4, translate, rotate, perspective, vec3
import math
import os
from pathlib import Path
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from memory_profiler import profile

# Initialize pygame and create window
pygame.init()
window_size = (800, 600)
window = pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)

# Create ModernGL context
ctx = moderngl.create_context(share=True)

# Enable depth testing to fix face rendering order
ctx.enable(moderngl.DEPTH_TEST)

# Shader program (vertex and fragment shader)
vertex_shader = """
#version 330
uniform mat4 view;
uniform mat4 projection;

in vec3 in_vert;
in vec2 in_texcoord;

// Per-instance data
in vec3 instance_pos;
in uint instance_rot;
in uint instance_scale;
in uint instance_index;

out vec2 frag_texcoord;

flat out uint blocksOnX;
flat out uint blocksOnY;
flat out uint currentInstanceIndexForFragShader;

mat4 rotation_matrix(vec3 rot) {
    float cx = cos(rot.x);
    float sx = sin(rot.x);
    float cy = cos(rot.y);
    float sy = sin(rot.y);
    float cz = cos(rot.z);
    float sz = sin(rot.z);

    return mat4(
        cy * cz, -cy * sz, sy, 0.0,
        sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy, 0.0,
        -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

void main() {
    mat4 rotation = rotation_matrix(vec3(0,0,0));
    
    // the if-statements makes it so the faces start growing out of one side.
    // ex. it will grow upwards, or grow right, instead of growing in both up and down at the same time
    vec3 scaled_vert = in_vert;
    
    uint bitsToShift = 16u;
    uint andValue = 0xFFFFu;
    
    blocksOnX = uint((instance_scale >> bitsToShift) & andValue);
    blocksOnY = uint((instance_scale >> 0u) & andValue);
    
    vec2 scaled_texcoord = in_texcoord;
    
    currentInstanceIndexForFragShader = instance_index;
    
    uint scaleX = ((blocksOnX - 1u) * 2u) + 1u;
    uint scaleY = ((blocksOnY - 1u) * 2u) + 1u;
    
    //texture_layer = texture_id;
    
    if(in_vert.x > 0){
        scaled_vert.x *= scaleX;
    }
    if(in_vert.y > 0){
        scaled_vert.y *= scaleY;
    }
    
    // switch the x, y, and z positions to correctly orientate the faces
    if(instance_rot == 0u){
    }
    if(instance_rot == 1u){
        scaled_vert.z += 1;
    }
    if(instance_rot == 2u){
        scaled_vert.yz = scaled_vert.zy;
        scaled_vert.y += 1;
    }
    if(instance_rot == 3u){
        scaled_vert.yz = scaled_vert.zy;
    }
    if(instance_rot == 4u){
        scaled_vert.xz = scaled_vert.zx;
    }
    if(instance_rot == 5u){
        scaled_vert.xz = scaled_vert.zx;
        scaled_vert.x += 1;
    }
    
    // TO DO: Get rid of the if statements because according to moderngl's docs (https://moderngl.readthedocs.io/en/5.10.0/the_guide/getting_started/triangles_draw/one_familiar_triangle.html)
    // GPUs don't like branching so "It is recommended to use formulas more often."
    
    mat4 model = rotation;
    model[3].xyz = instance_pos; // Add position
    frag_texcoord = scaled_texcoord;
    
    //gl_Position = projection * view * model * vec4(in_vert, 1);
    gl_Position = projection * view * model * vec4(scaled_vert, 1);
}

"""

fragment_shader = """
#version 430
in vec2 frag_texcoord;
out vec4 color;

uniform sampler2D tex;
uniform sampler2DArray texArray;

flat in uint currentInstanceIndexForFragShader;

flat in uint blocksOnX;
flat in uint blocksOnY;
//flat in uint textureLayerList;

layout(std430, binding = 0) buffer DataBuffer {
    int textureIdList[];
};

layout(std430, binding = 1) buffer OffsetBuffer {
    int offsets[];
};

void main() {
    // The 1-frag_texcoord flips it in that direction; flip to fix it
    uint blockX = uint(floor((1-frag_texcoord.x) * float(blocksOnX))); // Which block in the X direction
    uint blockY = uint(floor((1-frag_texcoord.y) * float(blocksOnY))); // Which block in the Y direction
    
    int start = offsets[currentInstanceIndexForFragShader];
    int end = offsets[currentInstanceIndexForFragShader+1];
    int lengthOfArr = end - start;
    
    uint layerIndex = (blockX * blocksOnY) + blockY;
    uint layer = uint( textureIdList[start + layerIndex] );
    
    // Scale the texcoords down to make each block actually have the range to properly fetch the texture (0 to 1)
    float blockTexcoordX = (frag_texcoord.x - (blockX/blocksOnX)) * blocksOnX;
    float blockTexcoordY = (frag_texcoord.y - (blockY/blocksOnY)) * blocksOnY;
    
    vec2 texcoord2 = vec2(blockTexcoordX, blockTexcoordY);
    
    color = texture( texArray, vec3(texcoord2, layer) );
}
"""

# Create shader program
program = ctx.program(
    vertex_shader=vertex_shader,
    fragment_shader=fragment_shader,
)

textureCache = {}
uniqueTextures = []

def generateTextureArray(ctx: moderngl.Context, texturePaths=[], textureWidth=16, textureHeight=16):
    if len(texturePaths) <= 0:
        raise IndexError("Cannot create texture array with no images!")
    
    textureArray = ctx.texture_array(
        (textureWidth, textureHeight, len(texturePaths)),
        4, # I'm assuming the textures are in RGBA format
        dtype='f1'
    )
    
    arrayData = b''
    
    for texturePath in texturePaths:
        textureImage = getTextureImage(texturePath, textureWidth, textureHeight, 0)
        
        arrayData += textureImage.tobytes()
    
    textureArray.write(arrayData)
    textureArray.filter = (moderngl.NEAREST, moderngl.NEAREST)
    
    return textureArray
  
@functools.cache
def getTextureImage(img_path, maxWidth=-1, maxHeight=-1, animationIndex=0):
    img = Image.open(img_path).convert("RGBA")  # Convert to RGB to ensure compatibility
    
    # Get the first 16x16 of the texture
    #maxWidth = 16
    #maxHeight = 16
    
    # Calculate the crop box for the top-right corner
    if maxWidth > 0 and maxHeight > 0:
        #xStart = animationIndex * maxWidth
        xStart = 0
        yStart = animationIndex * maxHeight
        crop_box = (xStart, yStart, xStart+maxWidth, yStart+maxHeight)
        img = img.crop(crop_box)
    
    return img

defaultTexturePath = "./missing_texture.png"
# Possibly do this on a per-chunk basis so we can purge it after loading a chunk
blockMap = {}
blockMapKeys = []
neighborOffsets = (
    (1, 0, 0),  # Right
    (-1, 0, 0), # Left
    (0, 1, 0),  # Up
    (0, -1, 0), # Down
    (0, 0, 1),  # Forward
    (0, 0, -1)  # Backward
)

# X - (neg x is right)
# Y - (neg y is down)
# Z - (neg z is forwards)
faceNameToFaceIndex = {
    "front":    0,
    "back":     1,
    "top":      2,
    "bottom":   3,
    "left":     4,
    "right":    5,
}

def isFaceFull(x, y, z, faceName):
    self = blockMap[(x, y, z)]
    
    mask = 0b000000
    if faceName == "front":  mask = 0b100000
    if faceName == "back":   mask = 0b010000
    if faceName == "top":    mask = 0b001000
    if faceName == "bottom": mask = 0b000100
    if faceName == "left":   mask = 0b000010
    if faceName == "right":  mask = 0b000001
    
    return (self["fullFaces"] & mask) != 0

def updateActiveTextures(x, y, z):
    global uniqueTextures
    self = blockMap[(x, y, z)]
    
    if "all" in self["texturePaths"]:
        theAllTexture = self["texturePaths"]["all"]
        self["activeTextures"] = {
            "front": theAllTexture,
            "back": theAllTexture,
            "top": theAllTexture,
            "bottom": theAllTexture,
            "left": theAllTexture,
            "right": theAllTexture,
        }
        
        if theAllTexture not in uniqueTextures: uniqueTextures.append(theAllTexture)
        
        blockMap[(x, y, z)] = self
        return
    
    for face in self["facesToRender"]:
        if face in self.texturePaths:
            texturePath = self.texturePaths[face]
            self["activeTextures"][face] = texturePath
            
            if texturePath not in uniqueTextures: uniqueTextures.append(texturePath)

def restoreFacesToRender(x, y, z):
    self = blockMap[(x, y, z)]
    
    self["facesToRender"] = {
        "front",
        "back",
        "top",
        "bottom",
        "left",
        "right",
    }
    
    blockMap[(x, y, z)] = self

def removeFace(x, y, z, faceName):
    self = blockMap[(x, y, z)]
    self["facesToRender"].discard(faceName)
    blockMap[(x, y, z)] = self

def removeFacesBasedOnNeighbors(x, y, z, neighbors: list[Block]):
    self = blockMap[(x, y, z)]
    if type(neighbors) == Block: neighbors = [neighbors]
    self["restoreFacesToRender"]()
    
    for neighbor in neighbors:
        #if self.uuid == neighbor.uuid: continue
        
        numberOfDiffsInPosition = 0
        diffX = neighbor["x"] - self["x"]
        diffY = neighbor["y"] - self["y"]
        diffZ = neighbor["z"] - self["z"]
        
        if diffX != 0: numberOfDiffsInPosition += 1
        if diffY != 0: numberOfDiffsInPosition += 1
        if diffZ != 0: numberOfDiffsInPosition += 1
        
        #if (diffX*diffX + diffY*diffY + diffZ*diffZ) > 2**2:
        #    continue
        
        if numberOfDiffsInPosition == 0:
            print(self["x"], self["y"], self["z"], neighbor["x"], neighbor["y"], neighbor["z"])
            raise RuntimeError("Hey! There are two blocks in the same space! This is not allowed, fix it?")
        elif numberOfDiffsInPosition > 1:
            continue # None of our faces are touching this block, so we can safely skip it
        
        # Make the full faces of neighbors opposite of what we're removing
        # because if our right face is touching, then we want to see if the LEFT face is
        # solid, not their right face
        
        if diffX == 1 and neighbor["isFaceFull"]("left"): # Block on our Right
            self["removeFace"]("right")
        if diffX == -1 and neighbor["isFaceFull"]("right"): # Block on our Left
            self["removeFace"]("left")
        
        if diffY == 1 and neighbor["isFaceFull"]("bottom"): # Block on Top
            self["removeFace"]("top")
        if diffY == -1 and neighbor["isFaceFull"]("top"): # Block on our Bottom
            self["removeFace"]("bottom")
        
        if diffZ == 1 and neighbor["isFaceFull"]("front"): # Block on Front
            self["removeFace"]("back")
        if diffZ == -1 and neighbor["isFaceFull"]("back"): # Block on our Back
            self["removeFace"]("front")
    
    #self.generateRenderingBuffersAndArrays() # Should happen in `removeFace`
    blockMap[(x, y, z)] = self

def Block(x=0, y=0, z=0, texturePaths: dict={"all": "./missing_texture.png"}, fullFaces=0b000000):
    out = {
        "fullFaces": fullFaces,
        "x": x,
        "y": y,
        "z": z,
        "texturePaths": texturePaths,
        "neighbors": 0,
        "activeTextures": {
            "front": defaultTexturePath,
            "back": defaultTexturePath,
            "top": defaultTexturePath,
            "bottom": defaultTexturePath,
            "left": defaultTexturePath,
            "right": defaultTexturePath,
        },
    }
    
    out["isFaceFull"] = lambda faceName: isFaceFull(x, y, z, faceName)
    out["updateActiveTextures"] = lambda: updateActiveTextures(x, y, z)
    out["restoreFacesToRender"] = lambda: restoreFacesToRender(x, y, z)
    out["removeFace"] = lambda faceName: removeFace(x, y, z, faceName)
    out["removeFacesBasedOnNeighbors"] = lambda neighbors: removeFacesBasedOnNeighbors(x, y, z, neighbors)
    
    for offset in neighborOffsets:
            newPos = (x - offset[0], y - offset[1], z - offset[2])
            if newPos in blockMap:
                out["neighbors"] += 1
                blockMap[newPos]["neighbors"] += 1
    
    blockMap[(x, y, z)] = out
    
    out["facesToRender"] = set()
    out["restoreFacesToRender"]()
    out["updateActiveTextures"]()
        
    return out

# Camera class
class Camera:
    def __init__(self, position, yaw, pitch, tilt):
        self.position = vec3(position)
        
        self.yaw = yaw
        self.pitch = pitch
        self.tilt = tilt
        
        self.fov = 45
        self.aspectRatio = window_size[0] / window_size[1]
        self.nearPlane = 0.1
        self.farPlane = 1000
        
        self.projection = perspective(self.fov, self.aspectRatio, self.nearPlane, self.farPlane)

    def get_view_matrix(self):
        # Start with an identity matrix
        view = mat4(1)
        
        # Apply rotations (yaw, pitch, tilt)
        view = rotate(view, np.radians(self.tilt), vec3(0, 0, 1))
        view = rotate(view, np.radians(self.pitch), vec3(1, 0, 0))
        view = rotate(view, np.radians(self.yaw), vec3(0, 1, 0))
        
        # Translate to the camera position
        view = translate(view, -self.position)
        
        return view

    def move_forward(self, speed):
        # Move forward relative to the yaw direction
        self.position.x += speed * math.sin(math.radians(self.yaw))
        self.position.z -= speed * math.cos(math.radians(self.yaw))

    def move_backward(self, speed):
        # Move backward relative to the yaw direction
        self.position.x -= speed * math.sin(math.radians(self.yaw))
        self.position.z += speed * math.cos(math.radians(self.yaw))

    def strafe_left(self, speed):
        # Move left relative to the yaw direction (perpendicular to forward)
        self.position.x -= speed * math.cos(math.radians(self.yaw))
        self.position.z -= speed * math.sin(math.radians(self.yaw))

    def strafe_right(self, speed):
        # Move right relative to the yaw direction (perpendicular to forward)
        self.position.x += speed * math.cos(math.radians(self.yaw))
        self.position.z += speed * math.sin(math.radians(self.yaw))

# Initialize camera
camera = Camera(position=(2, 2, 5), yaw=0, pitch=0, tilt=0)

# Set up the projection matrix for 3D rendering
program['projection'].write(camera.projection)

@functools.cache
def getFilesInDir(path):
    return {file.name: file for file in path.iterdir() if file.is_file()}

@functools.cache
def getFilesStartingWithString(path, string):
    path = Path(path)  # Use Pathlib for convenience
    files_in_dir = getFilesInDir(path)
    
    # Check exact match first
    target_file = f"{string}.png"
    if target_file in files_in_dir:
        return [target_file]
    
    # Check for override key
    if string in blockTextureOverrideKeys:
        return [blockTextureOverride[string]]
    
    # Find partial match
    for file_name in files_in_dir:
        if string in file_name:
            return [file_name]
    
    return []

specialBlocksToFullFaces = {
    "air": 0b000000,
}
blockTextureOverride = {
    "lava": "lava_still.png"
}
blockTextureOverrideKeys = tuple(blockTextureOverride.keys())

blocksToSkip = (
    "air",
)

chunks = {}

def loadSubChunkFromJson(subChunk):
    blocksAsIndices = subChunk[0] # 1 Sub Chunk is 16x16x16
    blockPalette = subChunk[1]
    startPosition = subChunk[2]
    
    chunkPos = [math.floor(startPosition[0] / 16), math.floor(startPosition[2] / 16)]
    chunkPosKey = f"{chunkPos[0]},{chunkPos[1]}"
    if chunkPosKey not in list(chunks.keys()):
        chunks[chunkPosKey] = []
    
    for y in range(16):
        for z in range(16):
            for x in range(16):
                blockIndex = x + (z * 16) + (y * 16 * 16)
                paletteIndex = blocksAsIndices[blockIndex]
                try:
                    blockFromPalette = blockPalette[paletteIndex]
                except IndexError as e:
                    print(paletteIndex, blockPalette, len(blockPalette))
                    raise e
                
                blockName = blockFromPalette["Name"]
                blockName = blockName.split("minecraft:")[1]
                #print(blockName)
                
                if blockName in blocksToSkip:
                    #print(f"Skipping {blockName}")
                    continue
                
                textureFolder = "./BlockTextures1_21_3"
                possibleFiles = getFilesStartingWithString(textureFolder, blockName)
                if len(possibleFiles) == 0: continue
                
                texturePath = f"{textureFolder}/{possibleFiles[0]}"
                textures = {
                    "all": texturePath
                }
                
                fullFaces = 0b111111
                if blockName in specialBlocksToFullFaces:
                    fullFaces = specialBlocksToFullFaces[blockName]
                
                position = (
                    x + startPosition[0],
                    y + startPosition[1],
                    z + startPosition[2],
                )
                
                # Maybe we can prematurely do the removing of nearby faces here to save time?
                newBlock = Block(
                    position[0],
                    position[1],
                    position[2],
                    textures,
                    fullFaces
                )
                chunks[chunkPosKey].append(newBlock)
    
    return True

def distributeUpdateBlockMap(blocks):
    global blockMap, blockMapKeys
    blockMap = {}
    
    def setBlockInBlockMap(block):
        global blockMap
        block[(block["x"], block["y"], block["z"])] = block
    
    with ThreadPoolExecutor() as executor:
        executor.map(setBlockInBlockMap, blocks)
    
    blockMapKeys = blockMap.keys()

def getNeighborsOfBlock(block):
    localBlockMap = blockMap
    localNeighborOffsets = neighborOffsets
    
    x, y, z = block["x"], block["y"], block["z"]
    neighbors = []
    for dx, dy, dz in localNeighborOffsets:
        neighbor = localBlockMap.get((x + dx, y + dy, z + dz))
        if neighbor is not None:
            neighbors.append(neighbor)
    
    return neighbors

def processBlockNeighbors(block):
    neighbors = getNeighborsOfBlock(block)
    return (block, neighbors)

def removeUnseenFaces(blocks):
    print("Starting to get neighbors list")
    for block in blocks:
        neighbors = processBlockNeighbors(block)[1]
        # Do a simple hack. assume that if we have six neighbors, we're not visible
        if len(neighbors) >= 6:
            block["facesToRender"] = set()
            continue
        else: continue
        
        block.removeFacesBasedOnNeighbors(neighbors)
    print("Done culling faces using thread pool 2")

def handleMovement(dt):
    # Handle keyboard input for camera movement
    keys = pygame.key.get_pressed()
    moveSpeed = 50 * dt  # movement speed
    rotationSpeed = 3000 * dt  # movement speed

    if keys[K_w]:  # Move forward relative to yaw
        camera.move_forward(moveSpeed)
    if keys[K_s]:  # Move backward relative to yaw
        camera.move_backward(moveSpeed)
    if keys[K_a]:  # Move left (strafe)
        camera.strafe_left(moveSpeed)
    if keys[K_d]:  # Move right (strafe)
        camera.strafe_right(moveSpeed)
    if keys[K_SPACE]:  # Move up
        camera.position.y += moveSpeed
    if keys[K_LSHIFT]:  # Move down
        camera.position.y -= moveSpeed

    # Rotate camera
    if keys[K_LEFT]:  # Rotate left
        camera.yaw -= rotationSpeed * dt
    if keys[K_RIGHT]:  # Rotate right
        camera.yaw += rotationSpeed * dt
    if keys[K_UP]:  # Rotate up
        camera.pitch -= rotationSpeed * dt
    if keys[K_DOWN]:  # Rotate down
        camera.pitch += rotationSpeed * dt

def genQuadVerticesWithScale(scale):
    quad_vertices = np.array([
        # Positions     # Texture coords
        0.5*scale,   0.5*scale, -0.5*scale,  0, 0,
        -0.5*scale,  0.5*scale, -0.5*scale,  1, 0,
        0.5*scale,  -0.5*scale, -0.5*scale,  0, 1,
        -0.5*scale, -0.5*scale, -0.5*scale,  1, 1,
    ], dtype='f4')
    return quad_vertices

quad_vertices = genQuadVerticesWithScale(1)
quad_indices = np.array([0, 1, 2, 1, 2, 3], dtype='i4')
quadVbo = ctx.buffer(quad_vertices.tobytes())
quadIbo = ctx.buffer(quad_indices.tobytes())

textureAndInstanceData = {}

bytesPerScaleValue = 1

def compressScale(scaleX, scaleY):
    ## To cover 1 extra block add 2 to the scale
    # Because our vertices are multiples of 0.5 adding 1 to the scale
    # will give us only 1/2 a block, so if we add 2 to the scale we'll get a full block 
    
    xShifted = (scaleX & 0xFFFF) << 16
    yShifted = (scaleY & 0xFFFF) << 0
    
    twoByteScale = xShifted | yShifted
    
    return twoByteScale

alreadyCheckedFaceToPosition = {}

def greedyMeshFrontBack(block, faceName, alreadyCheckedFaceToPosition):
    alreadyCheckedPositions = alreadyCheckedFaceToPosition[faceName]
    
    yOffset = 0
    while True:
        yOffset += 1
        positionToCheck = (block["x"], block["y"] + yOffset, block["z"])
        if positionToCheck in alreadyCheckedPositions: break # Face/Block is already in use for another bigger face
        
        if positionToCheck not in blockMapKeys: break # Not a block, breaks mesh & we cannot continue mesh with air block
        blockToCheck = blockMap[positionToCheck]
        
        if faceName not in blockToCheck["facesToRender"]: break # Block doesn't have the face we do, physically cannot continue mesh
        #if ourTexture != blockToCheck["activeTextures"][faceName]: break # Not the same texture, it breaks mesh
        # The face we're checking is active in both blocks
        
        alreadyCheckedFaceToPosition[faceName].append(positionToCheck)
        
        # And both blocks have the same texture on the face we're checking
    
    yOffset -= 1 # Remove one because we break when something is different, and by that point we've already added 1
    
    
    
    xOffset = 0
    doXLoop = True
    while doXLoop:
        xOffset += 1
        queueAlreadyChecked = []
        
        for y in range(0, yOffset+1):
            positionToCheck = (block["x"] + xOffset, block["y"] + y, block["z"])
            if positionToCheck in alreadyCheckedPositions:
                doXLoop = False
                break # Face/Block is already in use for another bigger face
            
            if positionToCheck not in blockMapKeys:
                doXLoop = False
                break # Not a block, breaks mesh & we cannot continue mesh with air block
            blockToCheck = blockMap[positionToCheck]
            
            if faceName not in blockToCheck["facesToRender"]:
                doXLoop = False
                break # Block doesn't have the face we do, physically cannot continue mesh
            # The face we're checking is active in both blocks
            
            queueAlreadyChecked.append(positionToCheck)
        
        if len(queueAlreadyChecked) == 0: break
        if doXLoop == False: break
        
        alreadyCheckedFaceToPosition[faceName].extend(queueAlreadyChecked)
    
    xOffset -= 1
    
    newFace = {
        "position": [block["x"], block["y"], block["z"]],
        "rotation": faceNameToFaceIndex[faceName],
        "scale": [1 + xOffset, 1 + yOffset],
        "texturesList": []
    }
    
    for xOff in range(0, xOffset+1):
        for yOff in range(0, yOffset+1):
            blockToGetInfo: Block = blockMap[(block["x"] + xOff, block["y"] + yOff, block["z"])]
            textureIndex = uniqueTextures.index(blockToGetInfo["activeTextures"][faceName])
            newFace["texturesList"].append(textureIndex)
    
    return newFace, alreadyCheckedFaceToPosition

def greedyMeshLeftRight(block, faceName, alreadyCheckedFaceToPosition):
    alreadyCheckedPositions = alreadyCheckedFaceToPosition[faceName]
    
    yOffset = 0
    while True:
        yOffset += 1
        positionToCheck = (block["x"], block["y"] + yOffset, block["z"])
        if positionToCheck in alreadyCheckedPositions: break # Face/Block is already in use for another bigger face
        
        if positionToCheck not in blockMapKeys: break # Not a block, breaks mesh & we cannot continue mesh with air block
        blockToCheck = blockMap[positionToCheck]
        
        if faceName not in blockToCheck["facesToRender"]: break # Block doesn't have the face we do, physically cannot continue mesh
        #if ourTexture != blockToCheck["activeTextures"][faceName]: break # Not the same texture, it breaks mesh
        # The face we're checking is active in both blocks
        
        alreadyCheckedFaceToPosition[faceName].append(positionToCheck)
        
        # And both blocks have the same texture on the face we're checking
    
    yOffset -= 1 # Remove one because we break when something is different, and by that point we've already added 1
    
    
    zOffset = 0
    doXLoop = True
    while doXLoop:
        zOffset += 1
        queueAlreadyChecked = []
        
        for y in range(0, yOffset+1):
            positionToCheck = (block["x"], block["y"] + y, block["z"] + zOffset)
            if positionToCheck in alreadyCheckedPositions:
                doXLoop = False
                break # Face/Block is already in use for another bigger face
            
            if positionToCheck not in blockMapKeys:
                doXLoop = False
                break # Not a block, breaks mesh & we cannot continue mesh with air block
            blockToCheck = blockMap[positionToCheck]
            
            if faceName not in blockToCheck["facesToRender"]:
                doXLoop = False
                break # Block doesn't have the face we do, physically cannot continue mesh
            """
            if ourTexture != blockToCheck["activeTextures"][faceName]:
                doXLoop = False
                break # Not the same texture, it breaks mesh
            """
            # The face we're checking is active in both blocks
            
            queueAlreadyChecked.append(positionToCheck)
        
        if len(queueAlreadyChecked) == 0: break
        if doXLoop == False: break
        
        alreadyCheckedFaceToPosition[faceName].extend(queueAlreadyChecked)
    
    zOffset -= 1
    
    newFace = {
        "position": [block["x"], block["y"], block["z"]],
        "rotation": faceNameToFaceIndex[faceName],
        "scale": [1 + zOffset, 1 + yOffset],
        "texturesList": [],
    }
    
    for zOff in range(0, zOffset+1):
        for yOff in range(0, yOffset+1):
            blockToGetInfo: Block = blockMap[(block["x"], block["y"] + yOff, block["z"] + zOff)]
            textureIndex = uniqueTextures.index(blockToGetInfo["activeTextures"][faceName])
            newFace["texturesList"].append(textureIndex)
    
    return newFace, alreadyCheckedFaceToPosition

def greedyMeshTopBottom(block, faceName, alreadyCheckedFaceToPosition):
    alreadyCheckedPositions = alreadyCheckedFaceToPosition[faceName]
    
    xOffset = 0
    while True:
        xOffset += 1
        positionToCheck = (block["x"] + xOffset, block["y"], block["z"])
        if positionToCheck in alreadyCheckedPositions: break # Face/Block is already in use for another bigger face
        
        if positionToCheck not in blockMapKeys: break # Not a block, breaks mesh & we cannot continue mesh with air block
        blockToCheck = blockMap[positionToCheck]
        
        if faceName not in blockToCheck["facesToRender"]: break # Block doesn't have the face we do, physically cannot continue mesh
        #if ourTexture != blockToCheck["activeTextures"][faceName]: break # Not the same texture, it breaks mesh
        # The face we're checking is active in both blocks
        
        alreadyCheckedFaceToPosition[faceName].append(positionToCheck)
        
        # And both blocks have the same texture on the face we're checking
    
    xOffset -= 1 # Remove one because we break when something is different, and by that point we've already added 1
    
    
    zOffset = 0
    doXLoop = True
    while doXLoop:
        zOffset += 1
        queueAlreadyChecked = []
        
        for x in range(0, xOffset+1):
            positionToCheck = (block["x"] + x, block["y"], block["z"] + zOffset)
            if positionToCheck in alreadyCheckedPositions:
                doXLoop = False
                break # Face/Block is already in use for another bigger face
            
            if positionToCheck not in blockMapKeys:
                doXLoop = False
                break # Not a block, breaks mesh & we cannot continue mesh with air block
            blockToCheck = blockMap[positionToCheck]
            
            if faceName not in blockToCheck["facesToRender"]:
                doXLoop = False
                break # Block doesn't have the face we do, physically cannot continue mesh
            # The face we're checking is active in both blocks
            
            queueAlreadyChecked.append(positionToCheck)
        
        if len(queueAlreadyChecked) == 0: break
        if doXLoop == False: break
        
        alreadyCheckedFaceToPosition[faceName].extend(queueAlreadyChecked)
    
    zOffset -= 1
    
    newFace = {
        "position": (block["x"], block["y"], block["z"]),
        "rotation": faceNameToFaceIndex[faceName],
        "scale": (1 + xOffset, 1 + zOffset),
        "texturesList": [],
    }
    
    for xOff in range(0, xOffset+1):
        for zOff in range(0, zOffset+1):
            blockToGetInfo: Block = blockMap[(block["x"] + xOff, block["y"], block["z"] + zOff)]
            textureIndex = uniqueTextures.index(blockToGetInfo["activeTextures"][faceName])
            newFace["texturesList"].append(textureIndex)
    
    return newFace, alreadyCheckedFaceToPosition

def greedyMeshBlocks(blocks: list[Block], faceName: Literal["None", "top", "bottom", "front", "back", "left", "right"]="None"):
    faces = []
    global alreadyCheckedFaceToPosition
    
    greedyMeshFrontBackFaceNames = ("front", "back")
    greedyMeshLeftRightFaceNames = ("left", "right")
    greedyMeshTopBottomFaceNames = ("top", "bottom")
    
    for block in blocks:
        if faceName not in block["facesToRender"]: continue
            
        if faceName not in alreadyCheckedFaceToPosition:
            alreadyCheckedFaceToPosition[faceName] = []
    
        alreadyCheckedPositions = alreadyCheckedFaceToPosition[faceName]
        if (block["x"], block["y"], block["z"]) in alreadyCheckedPositions: continue
    
        newFace = None
        args = (block, faceName, alreadyCheckedFaceToPosition)
    
        if faceName in greedyMeshTopBottomFaceNames:
            newFace, alreadyCheckedFaceToPosition = greedyMeshTopBottom(*args)
            pass
            
        elif faceName in greedyMeshFrontBackFaceNames:
            newFace, alreadyCheckedFaceToPosition = greedyMeshFrontBack(*args)
            pass
            
        elif faceName in greedyMeshLeftRightFaceNames:
            newFace, alreadyCheckedFaceToPosition = greedyMeshLeftRight(*args)
            pass
        
        
        if newFace != None:
            faces.append(newFace)
            
    return faces

def executeGreedyMeshOnBlock(specificFace):
    faces = []
    for chunkKey in list(chunks.keys()):
        faces.extend( greedyMeshBlocks(chunks[chunkKey], specificFace) )
    return faces

def distributeGreedyMeshAlgorithm():
    onlyUseSpecificFaceOptions = ("top", "bottom", "front", "back", "left", "right")
    
    listOfFaceResults = []
    for faceType in onlyUseSpecificFaceOptions:
        listOfFaceResults.extend( executeGreedyMeshOnBlock(faceType) )
    
    """
    with ThreadPoolExecutor() as executor:
        listOfFaceResults = list(executor.map(executeGreedyMeshOnBlock, onlyUseSpecificFaceOptions))
        print("Greedy mesh done, now applying faces into instance data")
    """
    
    global textureAndInstanceData
    textureAndInstanceData = {}
    
    extendInstanceDataFromFaceArray(listOfFaceResults)
    print("done turning faces into instance data!")
    
    """
    with ThreadPoolExecutor() as executor:
        lambdaTurnFaceIntoInstanceData = lambda face: extendInstanceDataFromFaceArray([face])
        executor.map(lambdaTurnFaceIntoInstanceData, faces)
    
        print("done turning faces into instance data!")
    """
    
    return len(listOfFaceResults)


# Rendering variables like buffers
allInstanceData = {
    "positions": [],
    "rotations": [],
    "scales": [],
    "textureLayerIds": [],
}

def extendInstanceDataFromFaceArray(faceArray):
    global allInstanceData
    
    for face in faceArray:
        rotX = face["scale"][0]
        rotY = face["scale"][1]
        
        allInstanceData["positions"].append(face["position"])
        allInstanceData["rotations"].append(face["rotation"])
        allInstanceData["scales"].append( compressScale(rotX, rotY) )
        allInstanceData["textureLayerIds"].append( face["texturesList"] )
    
    return len(allInstanceData["rotations"])

positionsBuffer = None
rotationsBuffer = None
scalesBuffer = None
textureIdsListBuffer = None
textureIdOffsetsBuffer = None
currentInstanceBuffer = None

fullCompleteInstanceDataCombined = None
numOfFaces = None

def generateBuffersIfNeeded():
    global  positionsBuffer, rotationsBuffer, scalesBuffer
    global textureIdsListBuffer, textureIdOffsetsBuffer, currentInstanceBuffer
    global fullCompleteInstanceDataCombined, numOfFaces
    
    if positionsBuffer == None:
        positionsList = allInstanceData["positions"]
        positionsBytes = np.array(positionsList).astype('f4').tobytes()
        positionsBuffer = ctx.buffer( positionsBytes )
    
    if rotationsBuffer == None:
        rotationsList = allInstanceData["rotations"]
        rotationsBytes = np.array(rotationsList).astype(np.uint32).tobytes()
        rotationsBuffer = ctx.buffer( rotationsBytes )
    
    if scalesBuffer == None:
        scalesList = allInstanceData["scales"]
        # I don't know why its making me use uint32 instead of uint16
        # But if I use uint16 some of the faces are 1/2 the size
        scalesBytes = np.array(scalesList).astype(np.uint32).tobytes()
        scalesBuffer = ctx.buffer( scalesBytes )
    
    if (textureIdsListBuffer == None or textureIdOffsetsBuffer == None or currentInstanceBuffer == None):
        textureIdListList = allInstanceData["textureLayerIds"]
        
        currentInstanceData = []
        for i in range(0, len(textureIdListList)): currentInstanceData.append(i)
        currentInstanceBytes = np.array(currentInstanceData).astype(np.uint32).tobytes()
        currentInstanceBuffer = ctx.buffer(currentInstanceBytes)
        
        
        flattened_data = [item for sublist in textureIdListList for item in sublist]
        offsets = [0]
        for sublist in textureIdListList:
            offsets.append(offsets[-1] + len(sublist))
        
        
        textureIdsListBuffer = ctx.buffer(np.array(flattened_data, dtype='i4').tobytes())
        textureIdOffsetsBuffer = ctx.buffer(np.array(offsets, dtype='i4').tobytes())
        
        textureIdsListBuffer.bind_to_storage_buffer(0)
        textureIdOffsetsBuffer.bind_to_storage_buffer(1)
    

timeElapsed = 0
clock = pygame.time.Clock()
def renderFrame():
    global timeElapsed, positionsBuffer, rotationsBuffer, scalesBuffer
    global textureIdsListBuffer, textureIdOffsetsBuffer, currentInstanceBuffer
    global fullCompleteInstanceDataCombined, numOfFaces, quadVbo
    
    dt = clock.tick(60)/1000
    fps = clock.get_fps()
    #print(fps, dt, camera.position)
    
    timeElapsed += dt
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            killRenderer()
            return False

    handleMovement(dt)

    # Calculate view matrix
    view = camera.get_view_matrix()
    program['view'].write(view)

    # Clear the screen (and set bg color)
    ctx.clear(13/255, 70/255, 168/255)
    
    if fullCompleteInstanceDataCombined == None:
        generateBuffersIfNeeded()
    
    if fullCompleteInstanceDataCombined == None:
        allPartsHere = (quadVbo != None and positionsBuffer != None and rotationsBuffer != None and scalesBuffer != None and currentInstanceBuffer != None)
        
        if allPartsHere:
            fullCompleteInstanceDataCombined = [
                (quadVbo, '3f 2f', 'in_vert', 'in_texcoord'),
                (positionsBuffer, '3f/i', 'instance_pos'),
                (rotationsBuffer, '1u/i', 'instance_rot'),
                (scalesBuffer, 'u/i', 'instance_scale'),
                (currentInstanceBuffer, 'u/i', "instance_index")
            ]
            # Don't need these anymore
            del quadVbo
            del positionsBuffer
            del rotationsBuffer
            del scalesBuffer
            del textureIdsListBuffer
            del textureIdOffsetsBuffer
            del currentInstanceBuffer
    
    if numOfFaces == None:
        numOfFaces = len(allInstanceData["rotations"])
    
    # https://moderngl.readthedocs.io/en/5.10.0/topics/buffer_format.html
    vao = ctx.vertex_array(
        program,
        fullCompleteInstanceDataCombined,
        quadIbo
    )
    """
    vao = ctx.vertex_array(
        program,
        [
            (quadVbo, '3f 2f', 'in_vert', 'in_texcoord'),
            (positionsBuffer, '3f/i', 'instance_pos'),
            (rotationsBuffer, '1u/i', 'instance_rot'),
            (scalesBuffer, 'u/i', 'instance_scale'),
            (currentInstanceBuffer, 'u/i', "instance_index")
        ],
        quadIbo
    )
    """
    
    vao.render(instances=numOfFaces)
    
    # Swap buffers
    pygame.display.flip()
    
    return True

def killRenderer():
    pygame.quit()