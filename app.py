from inpainter import Inpainter
import gradio as gr
import numpy as np
import io


inpainter = Inpainter()


def init(*args, **kwargs):
    rgb, depth = inpainter.init(*args, **kwargs)
    return rgb, (depth.clip(0, 64) * 1024).astype("uint16")


starter = gr.Interface(fn=init, inputs=[
    gr.inputs.Image(label="image", type="pil"),
    gr.inputs.Textbox(label="prompt"),
], outputs=[
    gr.components.Image(type="pil", label="image"),
    gr.components.Image(type="numpy", label="depth"),
])


def step(rgb, mask, depth, *args, **kwargs):
    if len(depth.shape) > 2:
        depth = depth[..., 0]
    rgb, depth = inpainter.step(rgb.convert("RGB"), mask, depth / 1024, *args, **kwargs)
    return rgb, (depth.clip(0, 64) * 1024).astype("uint16")


inpaint = gr.Interface(fn=step, inputs=[
    gr.inputs.Image(label="image", type="pil"),
    gr.inputs.Image(label="mask", type="pil"),
    gr.inputs.Image(label="depth", type="numpy"),
    gr.inputs.Textbox(label="prompt"),
], outputs=[
    gr.components.Image(type="pil", label="image"),
    gr.components.Image(type="numpy", label="depth"),
])
gr.TabbedInterface([starter, inpaint], ["Create a starting image", "Inpaint images"]).launch()
