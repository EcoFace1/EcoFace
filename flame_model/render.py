
import os, shutil
import cv2
import scipy
import tempfile
import numpy as np
from subprocess import call
import argparse
import sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # osmesa
import pyrender
import trimesh
import torch


class mesh_render:
    def __init__(self):
        #create scene
        camera_params = {'c': np.array([64, 64]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}
        frustum = {'near': 0.01, 'far': 3.0, 'height': 128, 'width': 128}

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
            color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

        return color[..., ::-1]


def render_sequence_meshes(sce,sequence_vertices, ft, mask = None):
    device=sequence_vertices.device
    ft=ft.cpu().numpy()
    sequence_vertices=sequence_vertices.detach().cpu().numpy()
    batch = sequence_vertices.shape[0]
    num_frames = sequence_vertices.shape[1]
    result = np.zeros((batch,num_frames,128,128,3))
    for b in range(batch):
        for v in range(num_frames):
            if mask is not None:
                if mask[b][v] == 0:
                    continue
            #pred_img = render_mesh_helper(sequence_vertices[b][v], ft)
            pred_img = sce.rend_scene(v=sequence_vertices[b][v], f=ft)
            pred_img = pred_img.astype(np.uint8)
            result[b][v]=pred_img
    return torch.FloatTensor(np.array(result)).to(device)

