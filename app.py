from torch import nn
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
# from matplotlib import pyplot as plt
import numpy as np
# from regex import D
# import scipy.spatial
# import numba as nb
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import torch

from typing import List
import logging
import trimesh
import cv2
# logger = logging.getLogger("numba");
# logger.setLevel(logging.ERROR)

import timm
from torchvision.transforms import Resize
# import safetensors
# from safetensors.torch import load_file
from torch import nn
import torch
# from matplotlib import pyplot as plt
from tqdm.auto import trange
from itertools import chain
from PIL import ImageOps
import scipy

from diffusers import StableDiffusionInpaintPipeline
# from transformers import set_seed
import random

from PIL import Image
import gc


def process_depth(dep, threshold=0.2, rescale=True, mask=None):
    depth = dep.copy()

    depth -= depth.mean()  # min()
    depth /= depth.std()  # max()
    depth = (depth + 3).clip(0., 100.)
    # depth = 1 / np.clip(depth, 0.2, 1)
    # 9 not available because it requires 8-bit
    blurred = cv2.medianBlur(depth, 5)
    maxd = cv2.dilate(blurred, np.ones((3, 3)))
    mind = cv2.erode(blurred, np.ones((3, 3)))
    edges = maxd - mind
    threshold = .05  # Better to have false positives
    pick_edges = edges > threshold
    # plt.imshow(pick_edges)
    # plt.colorbar()
    # plt.show()
    pick_edges = np.logical_or(
        pick_edges * 0,
        dep <= 1e-4
        # cv2.dilate((dep <= 1e-4).astype("uint8"), np.ones((3, 3))
    )
    if mask is not None:
        pick_edges = np.logical_and(pick_edges, np.logical_not(mask))
    # plt.imshow(pick_edges)
    # plt.colorbar()
    # plt.show()
    return (depth * 3 if rescale else dep), pick_edges


# @nb.jit
def make_mesh(im, depth, pick_edges):
    faces: List[np.ndarray] = []
    # grid = np.asarray(np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1])
    #                   )).transpose(1, 2, 0).reshape(-1, 2)[..., ::-1]
    grid = np.mgrid[0:im.shape[0], 0:im.shape[1]].transpose(1, 2, 0
                                                            ).reshape(-1, 2)[..., ::-1]
    flat_grid = grid[:, 1] * im.shape[1] + grid[:, 0]
    positions = np.concatenate(((grid - np.array(im.shape[:-1])[np.newaxis, :]
                                 / 2) / im.shape[1] * 2,
                                depth.flatten()[flat_grid][..., np.newaxis]),
                               axis=-1)
    positions[:, :-1] *= positions[:, -1:]
    positions[:, :2] *= -1
    colors = im.reshape(-1, 3)[flat_grid]

    # TODO: replace this with simple array expression
    y, x = (t.flatten() for t in np.mgrid[0:im.shape[0], 0:im.shape[1]])
    faces = np.concatenate((
        np.stack((y * im.shape[1] + x,
                 (y - 1) * im.shape[1] + x,
                  y * im.shape[1] + (x - 1)), axis=-1)
        [(~pick_edges.flatten()) * (x > 0) * (y > 0)],
        np.stack((y * im.shape[1] + x,
                   (y + 1) * im.shape[1] + x,
                   y * im.shape[1] + (x + 1)), axis=-1)
        [(~pick_edges.flatten()) * (x < im.shape[1] - 1) * (y < im.shape[0] - 1)]
    ))

    face_colors = np.asarray([colors[i[0]] for i in faces])
    return positions, faces, colors


def args_to_mat(tx, ty, tz, rx, ry, rz):
    mat = np.eye(4)
    mat[:3, :3] = scipy.spatial.transform.Rotation.from_euler(
        "XYZ", (rx, ry, rz)).as_matrix()
    mat[:3, 3] = tx, ty, tz
    return mat  # Create a renderer with the desired image size


# https://github.com/facebookresearch/pytorch3d/issues/84
class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        # (N, H, W, 4) RGBD image
        return torch.cat((images, fragments.zbuf), dim=-1)


class Renderer():
    def __init__(self, w=512, h=512, near=0.01, far=20., fov=89.99):
        self.w, self.h = w, h

        raster_settings = RasterizationSettings(
            image_size=self.h,  # 256,
            blur_radius=1e-6,
            faces_per_pixel=1,
            bin_size=None
        )
        batch = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cameras = FoVPerspectiveCameras(device=self.device, fov=fov)

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SimpleShader(device=self.device)
        )

    def __call__(self, vertices, faces, vert_colors):
        with torch.no_grad():
            textures = TexturesVertex(
                torch.from_numpy(vert_colors).unsqueeze(0).float().to(self.device))
            mesh = Meshes(torch.from_numpy(vertices).unsqueeze(0).float().to(self.device),
                          torch.from_numpy(faces).unsqueeze(0).float().to(self.device),
                          textures)
            images = self.renderer(mesh)
        rgb, d = images.detach().cpu().numpy()[0, ..., :3], images.detach().cpu().numpy()[0, ..., -1]
        return rgb, process_depth(d, rescale=False)


def process_mesh(img, depth, mask=None):
    dep, edges = process_depth(depth, rescale=False, mask=mask)
    vertices, faces, vert_colors = make_mesh(img, depth, edges)
    vertices = vertices.reshape(-1, 3)
    return vertices, faces, vert_colors


#@title Set up depth estimator
# !wget -c https://gist.githubusercontent.com/neverix/2422c1276380201777b32f416498b5b0/raw/4b6a08a388b8a2dfa504ae2f5356da5beac2afeb/maskgen.py
# !wget -c https://gist.githubusercontent.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0/raw/f417b71d7613ab410d7b2b715ee0e8541e559b36/midas_loss.py
# !wget -c https://huggingface.co/nev/masked-depth-estimator/resolve/main/masked_depth_estimator.safetensors



def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)




# def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
#     M = torch.sum(mask, (1, 2))

#     prediction = prediction.reshape((prediction.shape[0], -1))
#     target = target.reshape((target.shape[0], -1))
#     mask = mask.reshape((mask.shape[0], -1))
#     prediction, target = prediction[mask > 0.5], target[mask > 0.5]

#     pairs1, pairs2 = torch.randint(target.shape[0], (2, 1000))
#     diff1, diff2 = prediction[pairs1] - prediction[pairs2], target[pairs2] - target[pairs2]
#     image_loss = (1 + (diff1 * torch.tanh(diff2)).exp()).log()

#     return reduction(image_loss, M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    xs = []
    ys = []
    mask_x = torch.add(mask[:, :, 1:], mask[:, :, :-1]) > 0.5
    mask_y = torch.add(mask[:, 1:, :], mask[:, :-1, :]) > 0.5

    for diff in (prediction, target):
        grad_x = (diff[:, :, 1:] - diff[:, :, :-1])
        grad_y = (diff[:, 1:, :] - diff[:, :-1, :])
        xs.append(grad_x)
        ys.append(grad_y)
    
    diff_x = (xs[0] - xs[1]).pow(2) * mask_x
    diff_y = (ys[0] - ys[1]).pow(2) * mask_y
    # diff_x = (xs[0] - xs[1]).abs() * mask_x
    # diff_y = (ys[0] - ys[1]).abs() * mask_y
    image_loss = torch.sum(diff_x, (1, 2)) + torch.sum(diff_y, (1, 2))

    return reduction(image_loss, M)


# def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

#     M = torch.sum(mask, (1, 2))

#     diff = prediction - target
#     diff = torch.mul(mask, diff)

#     grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
#     mask_x = torch.add(mask[:, :, 1:], mask[:, :, :-1]) > 0.5
#     grad_x = torch.mul(mask_x, grad_x)

#     grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
#     mask_y = torch.add(mask[:, 1:, :], mask[:, :-1, :]) > 0.5
#     grad_y = torch.add(mask_y, grad_y)

#     image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

#     return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales
    
    def downscale(self, x, step):
        return nn.functional.interpolate(x.unsqueeze(1), scale_factor=1/step,
                                         mode="bicubic", antialias=True)[:, 0]
        # return x[:, ::step, ::step]
        # while step > 1:
        #     x = (x[:, 0::step, 0::step] +
        #          x[:, 0::step, 1::step] +
        #          x[:, 1::step, 0::step] +
        #          x[:, 1::step, 1::step]) / 4
        #     step //= 2
        # return x


    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(self.downscale(prediction, step), self.downscale(target, step),
                                   self.downscale(mask, step), reduction=self.__reduction)
            # total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                  #  mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, ground, target, mask):

        scale, shift = compute_scale_and_shift(prediction, ground, mask)
        self.__prediction_ssi = prediction  # scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(prediction, ground, mask)
        # total = 0
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, 1 - mask)
            # total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask * 0 + 1)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class MidasDepth(nn.Module):
    def __init__(self, model_type="DPT_Large",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 is_inpainting=False):
        super().__init__()
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval().requires_grad_(False)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        tfm = self.transform.transforms[1]
        self.w, self.h = tfm._Resize__width, tfm._Resize__height
        if self.w != self.h:
            raise ValueError
        rs = Resize((self.w, self.h))
        # self.transform.transforms[1] = lambda x: {"image": rs(x["image"].permute(0, 3, 1, 2))}
        self.transform.transforms[1] = lambda x: {"image": x["image"].permute(0, 3, 1, 2)}
        self.transform.transforms[2]._NormalizeImage__mean = torch.tensor(self.transform.transforms[2]._NormalizeImage__mean).to(self.device).unsqueeze(-1).unsqueeze(-1)
        self.transform.transforms[2]._NormalizeImage__std = torch.tensor(self.transform.transforms[2]._NormalizeImage__std).to(self.device).unsqueeze(-1).unsqueeze(-1)
        self.transform.transforms[3] = lambda x: x["image"]
        self.transform.transforms[4] = lambda x: x
        if is_inpainting:
            proj = torch.nn.Conv2d(6, 1024, kernel_size=(16, 16), stride=(16, 16))
            proj.weight.data[:, :3] = self.model.pretrained.model.patch_embed.proj.weight.data
            proj.weight.data[:, 3:] *= 0.01
            proj.bias.data = self.model.pretrained.model.patch_embed.proj.bias.data
            self.model.pretrained.model.patch_embed.proj = proj.to(device)
            self.transform.transforms[2]._NormalizeImage__mean = torch.cat((
                self.transform.transforms[2]._NormalizeImage__mean,
                self.transform.transforms[2]._NormalizeImage__mean[:1] * 0 + 0.5,
                self.transform.transforms[2]._NormalizeImage__mean[:1] * 0,
                self.transform.transforms[2]._NormalizeImage__mean[:1] * 0))[:6]
            self.transform.transforms[2]._NormalizeImage__std = torch.cat((
                self.transform.transforms[2]._NormalizeImage__std,
                self.transform.transforms[2]._NormalizeImage__std[:1] * 0 + 0.5,
                self.transform.transforms[2]._NormalizeImage__std[:1] * 0 + 2,
                self.transform.transforms[2]._NormalizeImage__std[:1] * 0 + 1))[:6]



    def forward(self, image):
        # if not isinstance(image, torch.cuda.FloatTensor):
            # image = torch.from_numpy(np.asarray(image)).to(self.device)
        # if (image > 1).any():
            # image = image / 255.   # .astype("float64") / 255.
        # with torch.inference_mode():
        batch = self.transform(image).to(self.device)
        prediction = self.model(batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[-3:-1],
            mode="bicubic",
            align_corners=False,
        )[:, 0]
        # prediction = prediction - prediction.min() + 1.5
        # prediction = 20 / prediction
        return prediction  # .squeeze()


def get_depth(img: Image, depth=None, depth_mask=None, lr=1, l2_coef=10, iterations=500, floor=1, true_floor=0.5):
    im = torch.from_numpy(np.asarray(img)).to(device).float() / 255.
    og_depth = og_depth_estimator(im.unsqueeze(0) * 255.)[0]
    d = og_depth
    d = (d - d.min()) / (d.max() - d.min()) * (10 - 3) + 3
    d = 30 / d
    # d = d.max() - d
    # d = d / d.max() * 15
    # d = d + 1.5
    if depth is None:
        return d.detach().cpu().numpy()
        depth_mask = np.asarray(img)[:, :, 0] * 0.
        depth = depth_mask.astype("float32") * 0
    depth_mask = (np.asarray(depth_mask) > 0).astype(np.uint8)
    depth_mask = cv2.erode(depth_mask, np.ones((3, 3)))
    if depth_mask.any():
        floor = max(true_floor, min(floor, depth[depth_mask > 0.5].min()))
    depth = torch.from_numpy(depth).to(device)
    depth_mask = torch.from_numpy(depth_mask).to(device).float()
    with torch.set_grad_enabled(True):
        # depth_mask = depth_mask == 0
        param = torch.nn.Parameter(torch.nan_to_num(d.clone().double().log()), requires_grad=True)
        optim = torch.optim.Adam([param], lr=lr)
        bar = trange(iterations)
        # img_dep[mask] = 0
        for _ in bar:
            # du = depth.clone()
            du = torch.nan_to_num(depth.clone().double().log())
            # du[depth_mask < 0.5] = nn.functional.softplus(param[depth_mask < 0.5] - floor, beta=0.25) + floor  # .exp()
            du[depth_mask < 0.5] = param[depth_mask < 0.5]
            # plt.imshow(du.detach().cpu().numpy())
            # plt.colorbar()
            # plt.show()
            # du = param
            loss = ScaleAndShiftInvariantLoss()(du.unsqueeze(0), torch.nan_to_num(depth.unsqueeze(0).double().log()),
                                                torch.nan_to_num(d.unsqueeze(0).double().log()), depth_mask.unsqueeze(0))
            loss.backward()
            optim.step()
            optim.zero_grad()
            bar.set_postfix(loss=loss.item())  # , c=c.item())
    return du.exp().float().detach().cpu().numpy()


class Scene3D(object):
    def __init__(self):
        self.vertices = np.empty((0, 3), dtype="float64")
        self.vertex_colors = np.empty((0, 3), dtype="float64")
        self.faces = np.empty((0, 3), dtype="uint64")
        self.renderer = Renderer()

    def next(self, mat, rgb, depth, msk, inplace=True):
        vertices, faces, vertex_colors = process_mesh(np.asarray(rgb) / 255., depth, msk)
        faces += len(self.vertices)
        vertices = np.concatenate((self.vertices, vertices))
        faces = np.concatenate((self.faces, faces))
        vertex_colors = np.concatenate((self.vertex_colors, vertex_colors))
        vertices = (
            np.concatenate((vertices, np.ones_like(vertices[..., -1:])), axis=-1)
            @ mat.T)[..., :3]
        im, (dep, mask) = self.renderer(vertices, faces, vertex_colors)
        if inplace:
            self.vertices, self.faces, self.vertex_colors = (
                vertices, faces, vertex_colors
            )
        im = Image.fromarray((im * 255).astype("uint8"))
        mask = Image.fromarray(mask == 0)
        return im, dep, mask


sd_model = "stabilityai/stable-diffusion-2-inpainting"  #@param {type: "string"}
# sd_model = "runwayml/stable-diffusion-inpainting"  #@param {type: "string"}
device = "cuda"
stable_pipe = StableDiffusionInpaintPipeline.from_pretrained(sd_model,
                                                            #  revision="fp16",
                                                             torch_dtype=torch.float16,
                                                             use_auth_token=True)
stable_pipe = stable_pipe.to(device)
stable_pipe.scheduler.set_timesteps(1000)
del stable_pipe.safety_checker
stable_pipe.safety_checker = None  # lambda clip_input, images: (images, [False for _ in images])


og_depth_estimator = MidasDepth()


torch.set_grad_enabled(False)
renderer = Renderer()



prompt = "Inside a fantasy dungeon; professional illustration; anime"  #@param  {type: "string"}
# start_image_url = "https://upload.wikimedia.org/wikipedia/commons/3/32/A_photograph_of_an_astronaut_riding_a_horse_2022-08-28.png"  #@param {type: "string"}
# download = lambda x: Image.open(io.BytesIO(requests.get(x).content)).convert("RGB")
# start_image = download(start_image_url)

seed = 17  #@param {type: "integer"}
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# set_seed(seed)
start_image = stable_pipe(prompt,
                          image=Image.new("RGB", (512, 512)),
                          mask_image=Image.new("L", (512, 512), 255)).images[0]
                          

scene = Scene3D()
im = start_image
dp = get_depth(im)
images = []
single = 4
frame_skip = 2.0
image, dep, msk = im, dp, dp * 0 + 1


# matrix = args_to_mat(0.1, 0, 0.0, 0.0, -0.1, 0)
matrix = args_to_mat(0.3, 0, 0.0, 0.0, 0.1, 0)
# matrix = args_to_mat(0.0, 0, 0.0, 0.0, -0.1, 0)
mat = scipy.linalg.fractional_matrix_power(matrix, frame_skip)

try:
    for _ in range(8):  # while True:
        gc.collect()
        torch.cuda.empty_cache()
        # plt.subplot(122)
        # plt.axis("off")
        # plt.imshow(im)
        # plt.subplot(121)
        # plt.imshow(dp)
        # plt.colorbar()
        # plt.show()
        # mat = args_to_mat(0.1, 0, 0.15, -0.1, 0.0, 0)
        # mat = scipy.linalg.fractional_matrix_power(mat, frame_skip)
        for i in trange(1, single + 1):
            # print(i / single)
            im, dp, mask = scene.next(scipy.linalg.fractional_matrix_power(mat, i / single) if i != single else mat,
                                    image,
                                    dep,
                                    msk,
                                    inplace=i == single)
            images.append(im.resize((256, 256)))
        # plt.subplot(122)
        # plt.axis("off")
        # plt.imshow(im)
        # plt.subplot(121)
        # plt.imshow(dp)
        # plt.colorbar()

        # plt.show()
        im = stable_pipe(prompt, im,
            ImageOps.invert(mask.convert("RGB")).convert("L")).images[0]
        # break
        dp = get_depth(im, depth=dp, depth_mask=np.asarray(mask).reshape(dp.shape))
        # mask = dp > 1
        image = im
        # To be used in a repeated generation loop
        # if (1 - mask).sum() == 0:
            # break
        msk = 1 - np.asarray(mask).reshape(dp.shape)
        mask = Image.fromarray(dp > 0)
        dep = dp
        # plt.imshow(msk)
        # plt.show()
except KeyboardInterrupt:
    # import imageio
    # imageio.mimsave("gorse.mp4", images)
    # from IPython.display import Video, display
    # display(Video("gorse.mp4"))
    pass

import imageio
imageio.mimsave("gorse.mp4", images)

