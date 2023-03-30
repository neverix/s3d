from inpainter import Inpainter
from myimage import Image
import gradio as gr
import numpy as np
import cv2


inpainter = Inpainter()


def init(*args, **kwargs):
    rgb, depth = inpainter.init(*args, **kwargs)
    return rgb, (depth.clip(0, 64) * 1024).astype("uint16")


starter = gr.Interface(fn=init, inputs=[
    gr.components.Image(label="rgb", type="pil"),
    gr.components.Textbox(label="prompt"),
], outputs=[
    gr.components.Image(type="pil", label="image"),
    gr.components.Image(type="numpy", label="depth"),
])


def step(rgb, mask, depth, *args, **kwargs):
    depth = cv2.imread(depth, -1)
    if len(depth.shape) > 2:
        depth = depth[..., 0]
    rgb, depth = inpainter.step(rgb.convert("RGB"), mask,
                                # depth / 255. * 64.,
                                depth / float(np.iinfo(depth.dtype).max) * 64.,
                                *args, **kwargs)
    return rgb, (depth.clip(0, 64) * 1024).astype("uint16")


inpaint = gr.Interface(fn=step, inputs=[
    gr.components.Image(label="rgb", type="pil", shape=(512, 512)),
    gr.components.Image(label="mask", type="pil", shape=(512, 512)),
    Image(label="depth", type="filepath"),  # , shape=(512, 512)),  # type="numpy", shape=(512, 512)),
    gr.components.Textbox(label="prompt"),
], outputs=[
    gr.components.Image(type="pil", label="image"),
    gr.components.Image(type="numpy", label="depth"),
])
gr.TabbedInterface([starter, inpaint], ["Create a starting image", "Inpaint images"]).launch(share=True)
