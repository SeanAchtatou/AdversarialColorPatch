import glob
import os
import sys
import image_process
import time

try:
    sys.path.append(glob.glob('../Carla_Simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480


def process_img(image,vehicle):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    status = image_process.apply_attack(image, vehicle)
    print(status)
    if status:
        apply(vehicle)
    return i3/255.0


def apply(vehicle):
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))


def simulatorLaunch():
    try:
        print("\033[1m\033[0;30;47m####################### CARLA SIMULATOR #######################\033[0m")
        print(" \033[93m\033[1m\033[4mServer\033[0m")
        print("\033[36m     Launching the server...\033[0m")
        os.startfile("..\Carla_Simulator\CarlaUE4.exe")
        print("\033[32m     Server launched.\033[0m")
        time.sleep(5)
    except:
        print("\033[91m     An error occured during the server launching.\033[0m")

    try:
        print(" \033[93m\033[1m\033[4mClient\033[0m")
        print("\033[36m     Launching client...")

        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()

        print("\033[36m     Creating and spawning the car...")
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('bmw')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)

        print("\033[36m     Adding the camera sensor and attaching it to the car...")
        blueprint = blueprint_library.find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
        blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
        blueprint.set_attribute('fov', '110')

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        print("\033[32m     All done.\033[0m")

        sensor.listen(lambda data: process_img(data,vehicle))


        input("\n\033[95mPress enter to stop the simulation...\033[0m")
        os.system("taskkill /f /im CarlaUE4.exe")

    except:
        print("An error occured.")





if __name__ == "__main__":
    simulatorLaunch()