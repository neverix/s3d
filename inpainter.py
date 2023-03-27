from diffusers import StableDiffusionInpaintPipeline
from typing import Optional, Union
from tqdm.auto import trange
from PIL import ImageOps
from PIL import Image
from torch import nn
import numpy as np
import torch
import cv2


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
        return nn.functional.interpolate(x.unsqueeze(1), scale_factor=1 / step,
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
        if self.device.type == "mps":
            self.device = torch.device("cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval().requires_grad_(False)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    @torch.no_grad()
    def forward(self, image):
        if torch.is_tensor(image):
            image = image.cpu().detach()
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        image = image.squeeze()
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


class Inpainter(nn.Module):
    def __init__(self,
                 depth_estimator=None,
#                 sd_model="stabilityai/stable-diffusion-2-inpainting",  # @param {type: "string"}
                  sd_model = "runwayml/stable-diffusion-inpainting",  #@param {type: "string"}
                 device=torch.device(  # "mps" if torch.backends.mps.is_available() else
                 "cuda" if torch.cuda.is_available() else "cpu"),
                 sd_device=torch.device("mps" if torch.backends.mps.is_available() else
                 "cuda" if torch.cuda.is_available() else "cpu"),
                 prompt=None):
        super().__init__()
        self.device = device
        self.sd_device = sd_device

        self.stable_pipe = StableDiffusionInpaintPipeline.from_pretrained(sd_model,
                                                                          #  revision="fp16",
                                                                          torch_dtype=torch.float16 if self.sd_device.type == "cuda:0" else torch.float32,
                                                                          use_auth_token=True,
                                                                          )
        self.stable_pipe = self.stable_pipe.to(self.sd_device)
        self.stable_pipe.enable_attention_slicing()
        del self.stable_pipe.safety_checker
        self.stable_pipe.safety_checker = None  # lambda clip_input, images: (images, [False for _ in images])

        if depth_estimator is None:
            depth_estimator = MidasDepth(device=self.device)
        self.depth_estimator = depth_estimator
        self.ssil = ScaleAndShiftInvariantLoss()

        self.prompt = prompt

    def _get_depth(self, img: Image.Image, depth=None, depth_mask=None, lr=1, l2_coef=10, iterations=500, floor=1,
                   true_floor=0.5):
        im = torch.from_numpy(np.asarray(img)).float().to(self.device) / 255.
        og_depth = self.depth_estimator(im.unsqueeze(0) * 255.)[0]
        d = og_depth
        d = (d - d.min()) / (d.max() - d.min()) * (10 - 3) + 3
        d = 30 / d
        # d = d.max() - d
        # d = d / d.max() * 15
        # d = d + 1.5
        if depth is None:
            return d.detach().cpu().numpy()
            # depth_mask = np.asarray(img)[:, :, 0] * 0.
            # depth = depth_mask.astype("float32") * 0
        depth_mask = (np.asarray(depth_mask) > 0).astype(np.uint8)
        depth_mask = cv2.erode(depth_mask, np.ones((3, 3)))
        if depth_mask.any():
            floor = max(true_floor, min(floor, depth[depth_mask > 0.5].min()))
        depth = torch.from_numpy(depth).float().to(self.device)
        depth_mask = torch.from_numpy(depth_mask).to(self.device).float()
        with torch.set_grad_enabled(True):
            # depth_mask = depth_mask == 0
            param = torch.nn.Parameter(torch.nan_to_num(d.clone().double().log()), requires_grad=True)
            optim = torch.optim.Adam([param], lr=lr)
            # bar = range(iterations)
            # img_dep[mask] = 0
            for _ in trange(iterations):
                # du = depth.clone()
                du = torch.nan_to_num(depth.clone().double().log())
                # du[depth_mask < 0.5] = nn.functional.softplus(param[depth_mask < 0.5] - floor, beta=0.25) + floor
                du[depth_mask < 0.5] = param[depth_mask < 0.5]
                # plt.imshow(du.detach().cpu().numpy())
                # plt.colorbar()
                # plt.show()
                # du = param
                loss = self.ssil(du.unsqueeze(0), torch.nan_to_num(depth.unsqueeze(0).double().log()),
                                 torch.nan_to_num(d.unsqueeze(0).double().log()),
                                 depth_mask.unsqueeze(0))
                loss.backward()
                optim.step()
                optim.zero_grad()
                # bar.set_postfix(loss=loss.item())  # , c=c.item())
        return du.exp().float().detach().cpu().numpy()

    @torch.no_grad()
    def init(self, image: Optional[Image.Image] = None, prompt: Optional[str] = None):
        """
        Get starting image/depth pair
        :param image: starting image (optional)
        :param prompt: prompt to generate the starting image (optional)
        """
        if image is None:
            if prompt is not None:
                self.prompt = prompt
            else:
                self.prompt = None
            image = self.stable_pipe(self.prompt,
                                     image=Image.new("RGB", (512, 512)),
                                     mask_image=Image.new("L", (512, 512), 255),
                                     num_inference_steps=25).images[0]
        depth = self._get_depth(image)
        return image, depth

    @torch.no_grad()
    def step(self, image: Image.Image, mask: Union[Image.Image, np.array], depth: np.array, prompt: Optional[str] = None):
        """
        Inpaint image and depth
        :param image: masked image (RGBA mode)
        :param depth: current depth
        :param prompt: prompt for demasking the image (optional)
        """
        image, mask = image.convert("RGB"), Image.fromarray(np.asarray(mask)).convert("L")
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = None  # lol. lmao even
        image = self.stable_pipe(self.prompt, image,
                                 # ImageOps only supports RGB
                                 ImageOps.invert(mask.convert("RGB")).convert("L"),
                                 num_inference_steps=25).images[0]
        # break
        depth = self._get_depth(image, depth=depth, depth_mask=np.asarray(mask).reshape(depth.shape) > 0)
        return image, depth


if __name__ == '__main__':
    inp = Inpainter(prompt="Inside a fantasy dungeon; professional illustration; anime")
    rgb, depth = inp.init()
    alpha = np.ones((512, 512))
    alpha[:128] = 0
    rgb.putalpha(Image.fromarray(alpha, mode="L"))
    rgb, depth = inp.step(rgb, depth)
    rgb.save("result.png")
    Image.fromarray(((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype("uint8"), mode="L").save("d.png")
