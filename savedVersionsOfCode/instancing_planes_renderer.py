from __future__ import annotations
import pygame
import moderngl
import numpy as np
from pygame.locals import *
from glm import mat4, translate, rotate, perspective, vec3
import math

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
in vec3 instance_rot;

out vec2 frag_texcoord;

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
    mat4 rotation = rotation_matrix(instance_rot);
    mat4 model = rotation;
    model[3].xyz = instance_pos; // Add position
    frag_texcoord = in_texcoord;
    gl_Position = projection * view * model * vec4(in_vert, 1);
}

"""

fragment_shader = """
#version 330
in vec2 frag_texcoord;
out vec4 color;

uniform sampler2D tex;

void main() {
    color = texture(tex, frag_texcoord);
}

"""

# Create shader program
program = ctx.program(
    vertex_shader=vertex_shader,
    fragment_shader=fragment_shader,
)

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
        self.farPlane = 100
        
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


def handleMovement(dt):
    # Handle keyboard input for camera movement
    keys = pygame.key.get_pressed()
    moveSpeed = 5 * dt  # movement speed
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


# Define quad vertices and texture coordinates
quad_vertices = np.array([
    # Positions     # Texture coords
    -0.5, -0.5, 0,  0, 0,
     0.5, -0.5, 0,  1, 0,
    -0.5,  0.5, 0,  0, 1,
     0.5,  0.5, 0,  1, 1,
], dtype='f4')

quad_indices = np.array([0, 1, 2, 1, 2, 3], dtype='i4')

# Create buffers
vbo = ctx.buffer(quad_vertices.tobytes())
ibo = ctx.buffer(quad_indices.tobytes())

# Load particle positions and rotations
num_particles = 10_000_000
positions = np.random.uniform(-10, 10, (num_particles, 3)).astype('f4')
rotations = np.random.uniform(0, 2 * np.pi, (num_particles, 3)).astype('f4')

instance_pos = ctx.buffer(positions.tobytes())
instance_rot = ctx.buffer(rotations.tobytes())

# Vertex Array Object
vao = ctx.vertex_array(
    program,
    [
        (vbo, '3f 2f', 'in_vert', 'in_texcoord'),
        (instance_pos, '3f/i', 'instance_pos'),
        (instance_rot, '3f/i', 'instance_rot'),
    ],
    ibo
)

from PIL import Image

texture_image = Image.open("texture.png").transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
texture = ctx.texture(texture_image.size, 4, texture_image.tobytes())
texture.use()
program['tex'] = 0


clock = pygame.time.Clock()
def renderFrame():
    dt = clock.tick(60)/1000
    fps = clock.get_fps()
    #print(fps, dt)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            killRenderer()
            return False

    handleMovement(dt)

    # Calculate view matrix
    view = camera.get_view_matrix()
    program['view'].write(view)

    # Clear the screen
    ctx.clear()
    
    vao.render(instances=num_particles)


    # Set model matrix (identity for now)
    model = mat4(1)
    #program['model'].write(model)

    # Swap buffers
    pygame.display.flip()
    #clock.tick(60)
    
    return True

def killRenderer():
    pygame.quit()