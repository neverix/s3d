from bpy_extras.view3d_utils import region_2d_to_origin_3d
from bpy_extras.view3d_utils import region_2d_to_vector_3d
import numpy as np
import threading
import requests
import tempfile
import random
import base64
import bmesh
import json
import bpy
import sys
import os
import io


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
    if mode == "depth":
        # bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
#        map = tree.nodes.new(type="CompositorNodeMapValue")
#        map.size = [1024]  # [1]  # /64]  # /1024]
#        map.use_min = False  # True
##        map.min = [0]
#        map.use_max = False  # True
##        map.max = [2 ** 16]
#        links.new(rl.outputs[2 if mode == "depth" else 1], map.inputs[0])
        map = tree.nodes.new(type="CompositorNodeMath")
        map.operation = "DIVIDE"
#        map.operation = "MULTIPLY"
        links.new(rl.outputs[2], map.inputs[0])
#        map.inputs[1].default_value = 64.0 * 1024.0
#        map.inputs[1].default_value = 1024.0
        map.inputs[1].default_value = 64.0 * 4.0
#        map.inputs[1].default_value = 1.0
        result = map
#        invert = tree.nodes.new(type="CompositorNodeInvert")
#        links.new(map.outputs[0], invert.inputs[1])
#        invert = tree.nodes.new(type="CompositorNodeMath")
#        invert.operation = "DIVIDE"
#        thresh.inputs[0].default_value = 1.0
#        links.new(map.outputs[0], invert.inputs[1])
#        result = invert
    elif mode == "alpha":
        bpy.context.scene.render.film_transparent = True
        thresh = tree.nodes.new(type="CompositorNodeMath")
        thresh.operation = "GREATER_THAN"
        links.new(rl.outputs[1], thresh.inputs[0])
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


def b64dec(image, save_dir=os.path.join(
           os.path.dirname(bpy.data.filepath), "generated_images")):
    assert image.startswith("data:image/png;base64,")
    image = base64.b64decode(image.partition("data:image/png;base64,")[-1])
    os.makedirs(save_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png",
                                     dir=save_dir,
                                     delete=False) as tf:
        open(tf.name, "wb").write(image)
        im = bpy.data.images.load(tf.name)
        te = np.asarray(im.pixels).reshape(im.size[1], im.size[0], -1)
    return im, te


def b64enc(image):
    open(os.path.join(os.path.dirname(bpy.data.filepath), "tmp.png"), "wb").write(image)
    result = "data:image/png;base64," + base64.b64encode(image).decode("utf-8")
    return result    


class S3D_PT_Panel(bpy.types.Panel):
    bl_label = "Depth completion"
    bl_idname = "S3D_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"
    
    def draw(self, context):
        rows = [self.layout.row() for _ in range(4)]
        rows[0].prop(bpy.context.scene.s3d_settings, "gradio_addr")
        rows[1].prop(bpy.context.scene.s3d_settings, "text")
        rows[2].prop(bpy.context.scene.s3d_settings, "run_for")
        rows[3].operator("s3d.complete")
        for row in rows:
            row.enabled = not CompleteDepth._running


class S3D_Batch_PT_Panel(bpy.types.Panel):
    bl_label = "Batched depth completion"
    bl_idname = "S3D_Batch_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"
    
    def draw(self, context):
        rows = [self.layout.row() for _ in range(5)]
        rows[0].prop(bpy.context.scene.s3d_settings, "gradio_addr")
        rows[1].prop(bpy.context.scene.s3d_settings, "script_path")
        rows[2].prop(bpy.context.scene.s3d_settings, "out_path")
        rows[3].prop(bpy.context.scene.s3d_settings, "run_for")
        rows[4].operator("s3d.complete_batch")
        for row in rows:
            row.enabled = not CompleteDepthBatch._running


class CompleteDepthBatch(bpy.types.Operator):
    """
    Completes the scene using 3D scene inpainting
    """
    
    bl_idname="s3d.complete_batch"
    bl_label="Complete Depth Batched"
    
    _step = 0
    _prompts = []
    _out_path = None
    _template = None
    _running = False
    
    def call_complete(self, context):
        bpy.context.scene.s3d_settings.text = self._prompts[self._step]
        window = context.window
        screen = window.screen
        views_3d = sorted(
                [a for a in screen.areas if a.type == 'VIEW_3D'],
                key=lambda a: (a.width * a.height))
        assert views_3d
        a = views_3d[0]
        # override
        o = {"window" : window,
             "screen" : screen,
             "area" : a,
             "space_data": a.spaces.active,
             "region" : a.regions[-1]
         }
        bpy.ops.s3d.complete(o)  # "INVOKE_DEFAULT")
        self._step += 1
    
    def modal(self, context, event):
#        self.__class__._running = False
        if not CompleteDepth._running or random.random() < 0.01:
            self.report({"INFO"}, str(CompleteDepth._running))
#        return {"FINISHED"}
        if event.type == "TIMER":
            if not CompleteDepth._running:
                self.report({"INFO"}, f"{self._step} {len(self._prompts)}")
                if self._step >= len(self._prompts):
                    self.report({"INFO"}, str(CompleteDepth._running))
                    self.__class__._running = False
                    return {"FINISHED"}
                sanitized_name = "".join((c if c.isalnum() else "_") for c in self._prompts[self._step - 1])
                bpy.ops.wm.save_as_mainfile(filepath=os.path.join(self._out_path, sanitized_name + ".blend"))
                bpy.ops.wm.open_mainfile(filepath=self._template)
                self.call_complete(context)
        return {"PASS_THROUGH"}
    
    def execute(self, context):
#        bpy.ops.s3d.complete("INVOKE_DEFAULT")
#        return {"FINISHED"}
#        if self.__class__._running:
#            return {"RUNNING_MODAL"}
        self.__class__._running = True
        self._out_path = bpy.context.scene.s3d_settings.out_path
        os.makedirs(self._out_path, exist_ok=True)
        # Let's hope no one names a prompt "  template"!
        self._template = os.path.join(self._out_path, "__template.blend")
        bpy.ops.wm.save_as_mainfile(filepath=self._template)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        self._step = 0
        self._prompts = open(bpy.context.scene.s3d_settings.script_path).read().split("\n")
#        return {"FINISHED"}
        self.call_complete(context)
        self.report({"INFO"}, str(CompleteDepth._running))
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self.__class__._running = False


class CompleteDepth(bpy.types.Operator):
    """
    Completes the scene using 3D scene inpainting
    """
    
    bl_idname="s3d.complete"
    bl_label="Complete Depth"
    
    _state = 0
    _step = 0
    _response = None
    _running = False
    
    def modal(self, context, event):
        try:
            if event.type == "TIMER":
                if self._state == 0:
                    rgb, depth = (b64enc(get_image(x)) for x in ("rgb", "depth"))
                    alpha = get_image("alpha")
                    self._mask = np.asarray(alpha.pixels).reshape(alpha.size[1], alpha.size[0], -1)
                    alpha = b64enc(alpha)
                    def fn(text, rgb, alpha, depth):
                        self._response = requests.post(os.path.join(
                            context.scene.s3d_settings.gradio_addr,
                            "run", "predict_1"
                        ), json=dict(data=[
                            rgb, alpha, depth,
                            text
                        ]), timeout=320 * 60).json()
                    threading.Thread(target=fn, args=(
                                     context.scene.s3d_settings.text,
                                     rgb, alpha, depth)).start()
                    bpy.context.scene.frame_set(self._step)
                    self._state = 1
                elif self._state == 1:
                    if self._response is not None:
                        self._state = 2
                elif self._state == 2:
                    rgb, depth = self._response["data"]
                    (rgb, _), depth = b64dec(rgb), b64dec(depth)[-1][..., 0] * 64.  # [..., 0].astype(np.float64)  # / 1024.
                    mask = depth > 1e-4

                    mesh = bpy.data.meshes.new("mesh")
                    obj = bpy.data.objects.new("image", mesh)
                    scene = context.scene
                    scene.collection.objects.link(obj)
                    bpy.context.view_layer.objects.active = obj
                    scene.render.resolution_x = 512
                    scene.render.resolution_y = 512
                    
                    mesh = bpy.context.object.data
                    bm = bmesh.new()
                    
                    vertices = {}
                    all_vertices = []

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


                    def try_get_vertex(bm, x, y):
                        if not (mask[y, x] or self._mask[y, x] > 1e-4):
                            return None
                        
                        if (x, y) in vertices:
                            return vertices[x, y]
                        
                        region = bpy.context.region
                        rv3d = bpy.context.region_data   

                        ray_origin, ray_vector = ray_cast(x, y)
                        _, center_vector = ray_cast(depth.shape[1] // 2, depth.shape[0] // 2)
                        z_current = np.dot(ray_vector, center_vector)
                        ray_vector = ray_vector / z_current
                        point = ray_origin + ray_vector * depth[-1 - y, x]
                        
                        vert = bm.verts.new(point)
                        vertices[x, y] = vert
                        all_vertices.append((x, y))
                        return vert

                    def try_make_triangle(bm, xys):
                        verts = []
                        for x, y in xys:
                            vert = try_get_vertex(bm, x, y)
                            if vert is None:
                                return None
                            verts.append(vert)
                        return bm.faces.new(verts)


                    for y in range(depth.shape[0] - 1):
                        for x in range(depth.shape[1] - 1):
                            a = try_make_triangle(bm, [(x, y), (x + 1, y), (x, y + 1)])
                            b = try_make_triangle(bm, [(x + 1, y), (x + 1, y + 1), (x, y + 1)])
                            if a is None and b is None:
                                try_make_triangle(bm, [(x, y), (x + 1, y + 1), (x, y + 1)])
                                try_make_triangle(bm, [(x, y), (x + 1, y), (x, y + 1)])
                    bm.verts.index_update()

                    mat = bpy.data.materials.new("map_material")
                    mat.use_nodes = True
                    obj.data.materials.append(mat)
                    for n in mat.node_tree.nodes:
                        if n.type == "OUTPUT_MATERIAL":
                            tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
                            tex.image = rgb
                            mat.node_tree.links.new(tex.outputs[0], n.inputs[0])
                        #    coord = mat.node_tree.nodes.new("ShaderNodeTexCoord")
                        #    coord.from_instancer = True
                        #    mat.node_tree.links.new(coord.outputs[2], tex.inputs[0])
                            break
                    uv_layer = bm.loops.layers.uv.new()
    #                tex_layer = bm.faces.layers.tex.new()
    #                uv_layer = bm.loops.layers.uv.get(uv_tex.name, False)
                    count = 0
                    for face in bm.faces:
    #                    face[tex_layer].image = rgb
                        for loop in face.loops:
                            # loop.vert.index
                            loop[uv_layer].uv = tuple(np.array([0.0, 1.0])
                                                + np.array([1.0, -1.0]) * np.asarray(
                                                all_vertices[loop.vert.index])
                                                / np.asarray(depth.shape)[::-1]
                                                )
                            count += 1

                    bm.to_mesh(mesh)
                    bm.free()
                    if self._step >= context.scene.s3d_settings.run_for - 1:
                        self.__class__._running = False
                        return {"FINISHED"}
                    else:
                        self._state = 0
                        self._step += 1
            return {"PASS_THROUGH"}
        except:
            self.__class__._running = False
            raise

    
    def execute(self, context):
#        if self.__class__._running:
#            return {"RUNNING_MODAL"}
        self.__class__._running = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        self._state = 0
        self._step = 0
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self.__class__._running = False


class S3DSettings(bpy.types.PropertyGroup):
    gradio_addr: bpy.props.StringProperty(
        name = "gradio address",
        default = "http://127.0.0.1:7860"
    )
    text: bpy.props.StringProperty(
        name = "text",
        default = "A fantasy dungeon"
    )
    run_for: bpy.props.IntProperty(
        name = "complete for",
        default = 1,
        min=1
    )
    script_path: bpy.props.StringProperty(
        name = "script path",
        default = "script.txt",
        subtype = "FILE_PATH"
    )
    out_path: bpy.props.StringProperty(
        name = "output path",
        default = "./out",
        subtype = "DIR_PATH"
    )


def register():
    bpy.utils.register_class(S3DSettings)
    bpy.types.Scene.s3d_settings = bpy.props.PointerProperty(type=S3DSettings)
    bpy.utils.register_class(CompleteDepth)
    bpy.utils.register_class(CompleteDepthBatch)
    bpy.utils.register_class(S3D_PT_Panel)
    bpy.utils.register_class(S3D_Batch_PT_Panel)

def unregister():
    bpy.utils.unregister_class(S3DSettings)
    del bpy.types.Scene.s3d_settings
    bpy.utils.unregister_class(CompleteDepth)
    bpy.utils.unregister_class(CompleteDepthBatch)
    bpy.utils.unregister_class(S3D_PT_Panel)
    bpy.utils.register_class(S3D_Batch_PT_Panel)


if __name__ == "__main__":
    register()
    # bpy.ops.s3d.complete()
