import numpy as np
import requests
import tempfile
import base64
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
        map.size = [1/1024]
        map.use_min = True
        map.min = [0]
        map.use_max = True
        map.max = [2 ** 16]
        links.new(rl.outputs[2], map.inputs[0])
        invert = tree.nodes.new(type="CompositorNodeInvert")
        links.new(map.outputs[0], invert.inputs[1])
        result = invert
        if mode == "alpha":
            thresh = tree.nodes.new(type="CompositorNodeMath")
            thresh.operation = "LESS_THAN"
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
        tf.write(image)
        tex = bpy.data.images.load(tf.name)
        tex = np.asarray(tex.pixels).reshape(tex.size[1], tex.size[0], -1)
    return tex


def b64enc(image):
    result = "data:image/png;base64," + base64.b64encode(image).decode("utf-8")
    return result    


result = requests.post("http://127.0.0.1:7860/run/predict_1", json=dict(data=[
    b64enc(get_image("rgb")),
    b64enc(get_image("alpha")),
    b64enc(get_image("depth")),
    "A fantasy dungeon"
])).json()
rgb, depth = result["data"]
rgb, depth = b64dec(rgb)[..., :3], b64dec(depth)[..., 0].astype(np.float64) / 1024.

