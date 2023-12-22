import os
from tqdm import tqdm

object_names = os.listdir("meshes/")

# object_names = ["armor"]

for o in tqdm(object_names):
    if '.html' in o:
        continue

    if 'sphere_nonhomogeneous' in o:
        continue
    mesh_names = os.listdir(f"meshes/{o}/")
    mesh_names = [s for s in mesh_names if ".obj" in s]

    assert(len(mesh_names) == 1)


    for m in mesh_names:
        src = f"meshes/{o}/{m}"
        dst = f"mesh_thumbnails/{m.replace('_remesh_lvl1', '').replace('obj', 'png')}"
        print(src, dst)
        os.system(f"space-thumbnails-cli.exe --input {src} {dst} --api vulkan")
