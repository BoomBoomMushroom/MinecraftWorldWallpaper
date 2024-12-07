# MinecraftWorldWallpaper
 
A voxel rendering engine to render minecraft worlds
Program flow:
- Read region file, decode it turn into NBT data
- Load blocks into classes and store in memory
- naive mesh remove unseen faces
- greedy mesh ALL faces
- calculate VAOs and store them (we don't need to make a VAO every frame b/c the world doesn't change)

We can greedy mesh everything because all the textures are rendered from a texture array and not sent perface!
