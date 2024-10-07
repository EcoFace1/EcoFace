import os
import sys
import tqdm
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyrender
import trimesh
import cv2
import argparse

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # osmesa
class mesh_render:
    def __init__(self):
        #create scene
        camera_params = {'c': np.array([128, 128]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}
        frustum = {'near': 0.01, 'far': 3.0, 'height': 256, 'width': 256}

        self.primitive_material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode='BLEND',
            baseColorFactor=[0.3, 0.3, 0.3, 1.0],
            metallicFactor=0.8,
            roughnessFactor=0.8
        )
        #rgb_per_v = None
        #tri_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=rgb_per_v)
        #render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

        self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])  # [0, 0, 0] black,[255, 255, 255] white

        camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                        fy=camera_params['f'][1],
                                        cx=camera_params['c'][0],
                                        cy=camera_params['c'][1],
                                        znear=frustum['near'],
                                        zfar=frustum['far'])

        #scene.add(render_mesh, pose=np.eye(4))

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, -0.02,2.9])
        self.scene.add(camera, pose=[[1, 0, 0, 0],
                                [0, 1, 0, -0.02],
                                [0, 0, 1, 2.9],
                                [0, 0, 0, 1]])

        angle = np.pi / 6.0
        pos = camera_pose[:3, 3]
        light_color = np.array([1., 1., 1.])
        light = pyrender.DirectionalLight(color=light_color, intensity=2.0)

        light_pose = np.eye(4)
        light_pose[:3, 3] = pos
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        self.flags = pyrender.RenderFlags.SKIP_CULL_FACES
        
        self.r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
    def rend_scene(self,v,f):
        rgb_per_v = None
        tri_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.primitive_material, smooth=True)
        mesh_node=pyrender.Node(mesh=render_mesh,matrix=np.eye(4))
        try:
            self.scene.add_node(mesh_node)
            color, _ = self.r.render(self.scene, flags=self.flags)
            self.scene.remove_node(mesh_node)
        except:
            print('pyrender: Failed rendering frame')
            color = np.zeros((256, 256, 3), dtype='uint8')

        return color[..., ::-1]


from flame_model.flame import FlameHead
model_flame=FlameHead(expr_params=50,shape_params=100)


#load pyrender
sce=mesh_render()
ft=model_flame.faces

def get_render(arg):
    save_path=os.path.join(arg.result_path,'pic')
    os.makedirs(save_path, exist_ok=True)

    npy_data = np.load(arg.npy_path, allow_pickle=True)

    for id in range(npy_data.shape[0]):
        pred_img = sce.rend_scene(v=npy_data[id], f=ft.cpu().numpy())
        cv2.imwrite(f'{save_path}/{id}.png', pred_img)

    # to mp4
    cmd = f"ffmpeg -framerate {arg.fps} -i {save_path}/%d.png -i {arg.audio_path} {os.path.join(arg.result_path,'demo.mp4')}"
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description='Ours model')
    parser.add_argument("--model_name", type=str, default="rav_hdtf")
    parser.add_argument("--fps", type=float, default=25, help='frame rate')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--npy_path", type=str, default="demo/result/test.npy", help='path of the input npy')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--audio_path", type=str, default="demo/test.wav", help='path of the predictions')
    args = parser.parse_args()

    get_render(args)


if __name__ == "__main__":
    main()