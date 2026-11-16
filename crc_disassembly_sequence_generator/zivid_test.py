import zivid
import matplotlib.pyplot as plt
import numpy as np
#import keyboard
import sys
sys.path.insert(0,r'C:\Program Files\Zivid\bin')

# enable ip4 protocol using static ip with the following settings for camera with default config
# IP address: 172.28.60.2
# Subnet mask: 255.255.255.0
# for further information visit
# https://support.zivid.com/en/latest/getting-started/software-installation/network-configuration/static-network-configuration-pc.html

n_captures : int = 1
save_path : str = r'C:\Users\MaH_Kretschmer\Desktop\zivid_captures'
def main():
    app = zivid.Application()
    camera = app.connect_camera() #physical camera with static ip
    #camera = app.create_file_camera("C:/ProgramData/Zivid/FileCameraZivid2M70.zfc") #emulated camera from sample data
    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])

    for i in range(n_captures):
        frame = camera.capture(settings)
        xyz = frame.point_cloud().copy_data("xyz")  # Get point coordinates as [Height,Width,3] float array
        xyz = np.nan_to_num(xyz).reshape(-1, 3)
        rgba = frame.point_cloud().copy_data("rgba")  # Get point colors as [Height,Width,4] uint8 array
        visualize_pt_cloud(xyz, 'Zivid Emulation Snapshot')
        print(f"Saving Snapshot {i+1} in directory: {save_path}")
        frame.save(save_path + "/capture_" + str(i+1) + ".zdf")
        print("Press space to continue.")
        #keyboard.wait("space")

def visualize_pt_cloud(points, title, markersize=2):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)

    # Add 3d axes
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=markersize)

    # Setting the labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show(block=True)

if __name__ == '__main__':
    main()