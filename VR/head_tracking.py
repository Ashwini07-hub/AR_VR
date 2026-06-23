import sys
import math
import time
import board
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX

# Initialize I2C and LSM6DSO Sensor using the correct subclass
try:
    i2c = board.I2C()
    sensor = LSM6DSOX(i2c, address=0x6b)
    print("LSM6DSO VR Headset Camera Initialized Successfully!")
except Exception as e:
    print(f"Sensor connection error: {e}")
    sys.exit()

# 3D Cube vertices and edges
vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)
edges = (
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,7), (3,6)
)

def DrawCube():
    glBegin(GL_LINES)
    glColor3f(0.0, 1.0, 0.0) # Bright Green lines
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("True VR Headset Head-Tracking")

    # Enable depth testing to make sure 3D rendering functions properly
    glEnable(GL_DEPTH_TEST)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    roll, pitch, yaw = 0.0, 0.0, 0.0
    last_time = time.time()

    print("True VR Camera mode running... Move the IMU to look around.")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        # Time difference calculation
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Read Sensor data
        ax, ay, az = sensor.acceleration 
        gx, gy, gz = sensor.gyro         

        # Convert to degrees/s
        gx = math.degrees(gx)
        gy = math.degrees(gy)
        gz = math.degrees(gz)

        # Calculate values
        accel_roll = math.atan2(ay, az) * 57.2958
        accel_pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2)) * 57.2958

        # Complementary Filter
        roll = 0.98 * (roll + gx * dt) + 0.02 * accel_roll
        pitch = 0.98 * (pitch + gy * dt) + 0.02 * accel_pitch
        yaw += gz * dt  

        # --- 3D GRAPHICS RENDER ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity() # Clear previous frame matrix modifications

        # 1. First, move the camera back so we aren't sitting right inside the center of the cube
        glTranslatef(0.0, 0.0, -6.0)

        # 2. Second, rotate the entire world frame in reverse to match head orientation
        glRotatef(-pitch, 1, 0, 0) 
        glRotatef(-yaw, 0, 1, 0)   
        glRotatef(-roll, 0, 0, 1)  

        # 3. Draw our structural cube asset
        DrawCube()

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()