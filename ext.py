from bpy_extras.view3d_utils import region_2d_to_origin_3d
from bpy_extras.view3d_utils import region_2d_to_vector_3d
import numpy as np
import requests
import tempfile
import base64
import bmesh
import json
import bpy
import os


def get_image(mode="rgb"):
    if mode not in ("rgb", "depth", "alpha"):
        raise ValueError(f"Invalid image mode: `{mode}`")
    # Set up rendering
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    links = tree.links
    bpy.context.scene.render.image_settings.color_depth = "16"
    # clear default nodes
    # TODO push old nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')
    if mode in ("depth", "alpha"):
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        map = tree.nodes.new(type="CompositorNodeMapValue")
        map.size = [1024]  # [1]  # /64]  # /1024]
        map.use_min = False  # True
        map.min = [0]
        map.use_max = False  # True
        map.max = [2 ** 16]
        links.new(rl.outputs[2], map.inputs[0])
        invert = tree.nodes.new(type="CompositorNodeInvert")
        links.new(map.outputs[0], invert.inputs[1])
        result = invert
        if mode == "alpha":
            thresh = tree.nodes.new(type="CompositorNodeMath")
            thresh.operation = "GREATER_THAN"
            links.new(result.outputs[0], thresh.inputs[0])
            thresh.inputs[1].default_value = 0.1
            result = thresh
    elif mode == "rgb":
        result = rl
    else:
        1000/0
    finalOutput = tree.nodes.new(type="CompositorNodeComposite")
    link = links.new(result.outputs[0], finalOutput.inputs[0])
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        fn = tf.name  # os.path.join(os.path.dirname(bpy.data.filepath), f"{'depth' if is_depth else 'rgb'}.png")
        bpy.context.scene.render.filepath = fn
        bpy.ops.render.render(write_still=True)
        image = open(fn, "rb").read()
    return image


def b64dec(image):
    assert image.startswith("data:image/png;base64,")
    image = base64.b64decode(image.partition("data:image/png;base64,")[-1])
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        open(tf.name, "wb").write(image)
        tex = bpy.data.images.load(tf.name)
        tex = np.asarray(tex.pixels).reshape(tex.size[1], tex.size[0], -1)
    return tex


def b64enc(image):
    open(os.path.join(os.path.dirname(bpy.data.filepath), "depth.png"), "wb").write(image)
    result = "data:image/png;base64," + base64.b64encode(image).decode("utf-8")
    return result    


result = requests.post("http://127.0.0.1:7860/run/predict_1", json=dict(data=[
    b64enc(get_image("rgb")),
    b64enc(get_image("alpha")),
    b64enc(get_image("depth")),
    "A fantasy dungeon"
])).json()
#result = requests.post("http://127.0.0.1:7860/run/predict", json=dict(data=[
#    b64enc(get_image("rgb")),
#    "A grey cube"
#])).json()x
rgb, depth = result["data"]
rgb, depth = b64dec(rgb)[..., :3], b64dec(depth)[..., 0] * 64  # [..., 0].astype(np.float64)  # / 1024.
mask = depth > 1e-4

mesh = bpy.data.meshes.new("mesh")
obj = bpy.data.objects.new("image", mesh)
scene = bpy.context.scene
scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
scene.render.resolution_x = 512
scene.render.resolution_y = 512

mesh = bpy.context.object.data
bm = bmesh.new()

vertices = {}

def ray_cast(x, y):
    cam = bpy.context.scene.camera  # bpy.data.objects["camera"]
    
    ## set view mode to 3D to have all needed variables available
    #bpy.context.area.type = "VIEW_3D"

    # get vectors which define view frustum of camera
    frame = [cam.matrix_world.normalized() @ u
             for u in cam.data.view_frame(scene=bpy.context.scene)]
    topRight = frame[0]
    bottomRight = frame[1]
    bottomLeft = frame[2]
    topLeft = frame[3]
    
    # number of pixels in X/Y direction
    resolutionX = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
    resolutionY = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))

    # interpolate
    x, y = x / resolutionX, y / resolutionY
    origin = cam.matrix_world.translation
    base = topLeft + (topRight - topLeft) * x    
    end = bottomLeft + (bottomRight - bottomLeft) * x    
    target = base + (end - base) * y
    return origin, (target - origin) / np.linalg.norm(target - origin)

def try_get_vertex(x, y):
    if not mask[y, x]:
        return None
    
    if (x, y) in vertices:
        return vertices[x, y]
    
    region = bpy.context.region
    rv3d = bpy.context.region_data   

    ray_origin, ray_vector = ray_cast(x, y)
    point = ray_origin + ray_vector * depth[y, x]
    
    vert = bm.verts.new(point)
    vertices[x, y] = vert
    return vert

def try_make_triangle(xys):
    verts = []
    for x, y in xys:
        vert = try_get_vertex(x, y)
        if vert is None:
            return None
        verts.append(vert)
    return bm.faces.new(verts)

for y in range(depth.shape[0] - 1):
    for x in range(depth.shape[1] - 1):
        a = try_make_triangle([(x, y), (x + 1, y), (x, y + 1)])
        b = try_make_triangle([(x + 1, y), (x + 1, y + 1), (x, y + 1)])
        if a is None and b is None:
            try_make_triangle([(x, y), (x + 1, y + 1), (x, y + 1)])
            try_make_triangle([(x, y), (x + 1, y), (x, y + 1)])

bm.to_mesh(mesh)  
bm.free()
