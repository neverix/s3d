from copy import deepcopy
from pathlib import Path
from PIL import Image as _Image  # using _ to minimize namespace pollution
import numpy as np  
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type
from typing_extensions import Literal
import tempfile
import base64
from gradio.events import (
    Blurrable,
    Changeable,
    Clearable,
    Clickable,
    Editable,
    EventListener,
    Playable,
    Releaseable,
    Streamable,
    Submittable,
    Uploadable,
)
from gradio.interpretation import NeighborInterpretable, TokenInterpretable
from gradio.layouts import Column, Form, Row
from gradio.processing_utils import TempFileManager
from gradio.serializing import (
    FileSerializable,
    ImgSerializable,
    JSONSerializable,
    Serializable,
    SimpleSerializable,
)
from gradio.documentation import document, set_documentation_group
from gradio.components import IOComponent, _Keywords, Component
from gradio import media_data, processing_utils, utils


@document("style")
class Image(
    Editable,
    Clearable,
    Changeable,
    Streamable,
    Uploadable,
    IOComponent,
    ImgSerializable,
    TokenInterpretable,
):
    """
    Creates an image component that can be used to upload/draw images (as an input) or display images (as an output).
    Preprocessing: passes the uploaded image as a {numpy.array}, {PIL.Image} or {str} filepath depending on `type` -- unless `tool` is `sketch` AND source is one of `upload` or `webcam`. In these cases, a {dict} with keys `image` and `mask` is passed, and the format of the corresponding values depends on `type`.
    Postprocessing: expects a {numpy.array}, {PIL.Image} or {str} or {pathlib.Path} filepath to an image and displays the image.
    Examples-format: a {str} filepath to a local file that contains the image.
    Demos: image_mod, image_mod_default_image
    Guides: Gradio_and_ONNX_on_Hugging_Face, image_classification_in_pytorch, image_classification_in_tensorflow, image_classification_with_vision_transformers, building_a_pictionary_app, create_your_own_friends_with_a_gan
    """

    def __init__(
        self,
        value: str | _Image.Image | np.ndarray | None = None,
        *,
        shape: Tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "upload",
        tool: str | None = None,
        type: str = "numpy",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        brush_radius: int | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: A PIL Image, numpy array, path or URL for the default value that Image component is going to take. If callable, the function will be called whenever the app loads to set the initial value of the component.
            shape: (width, height) shape to crop and resize image to; if None, matches input image size. Pass None for either width or height to only crop and resize the other.
            image_mode: "RGB" if color, or "L" if black and white.
            invert_colors: whether to invert the image as a preprocessing step.
            source: Source of image. "upload" creates a box where user can drop an image file, "webcam" allows user to take snapshot from their webcam, "canvas" defaults to a white image that can be edited and drawn upon with tools.
            tool: Tools used for editing. "editor" allows a full screen editor (and is the default if source is "upload" or "webcam"), "select" provides a cropping and zoom tool, "sketch" allows you to create a binary sketch (and is the default if source="canvas"), and "color-sketch" allows you to created a sketch in different colors. "color-sketch" can be used with source="upload" or "webcam" to allow sketching on an image. "sketch" can also be used with "upload" or "webcam" to create a mask over an image and in that case both the image and mask are passed into the function as a dictionary with keys "image" and "mask" respectively.
            type: The format the image is converted to before being passed into the prediction function. "numpy" converts the image to a numpy array with shape (width, height, 3) and values from 0 to 255, "pil" converts the image to a PIL image object, "filepath" passes a str path to a temporary file containing the image.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to upload and edit an image; if False, can only be used to display images. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            streaming: If True when used in a `live` interface, will automatically stream webcam feed. Only valid is source is 'webcam'.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            mirror_webcam: If True webcam will be mirrored. Default is True.
            brush_radius: Size of the brush for Sketch. Default is None which chooses a sensible default
        """
        self.brush_radius = brush_radius
        self.mirror_webcam = mirror_webcam
        valid_types = ["numpy", "pil", "filepath"]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        self.shape = shape
        self.image_mode = image_mode
        valid_sources = ["upload", "webcam", "canvas"]
        if source not in valid_sources:
            raise ValueError(
                f"Invalid value for parameter `source`: {source}. Please choose from one of: {valid_sources}"
            )
        self.source = source
        if tool is None:
            self.tool = "sketch" if source == "canvas" else "editor"
        else:
            self.tool = tool
        self.invert_colors = invert_colors
        self.test_input = deepcopy(media_data.BASE64_IMAGE)
        self.streaming = streaming
        if streaming and source != "webcam":
            raise ValueError("Image streaming only available if source is 'webcam'.")

        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            value=value,
            **kwargs,
        )
        TokenInterpretable.__init__(self)

    def get_config(self):
        return {
            "image_mode": self.image_mode,
            "shape": self.shape,
            "source": self.source,
            "tool": self.tool,
            "value": self.value,
            "streaming": self.streaming,
            "mirror_webcam": self.mirror_webcam,
            "brush_radius": self.brush_radius,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
        brush_radius: int | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "brush_radius": brush_radius,
            "__type__": "update",
        }
        return IOComponent.add_interactive_to_config(updated_config, interactive)

    def _format_image(
        self, im: _Image.Image | None
    ) -> np.ndarray | _Image.Image | str | None:
        """Helper method to format an image based on self.type"""
        if im is None:
            return im
        fmt = im.format
        if self.type == "pil":
            return im
        elif self.type == "numpy":
            return np.array(im)
        # elif self.type == "filepath":
        #     file_obj = tempfile.NamedTemporaryFile(
        #         delete=False,
        #         suffix=("." + fmt.lower() if fmt is not None else ".png"),
        #     )
        #     im.save(file_obj.name)
        #     return file_obj.name
        else:
            raise ValueError(
                "Unknown type: "
                + str(self.type)
                + ". Please choose from: 'numpy', 'pil', 'filepath'."
            )

    def generate_sample(self):
        return deepcopy(media_data.BASE64_IMAGE)

    def preprocess(
        self, x: str | Dict[str, str]
    ) -> np.ndarray | _Image.Image | str | Dict | None:
        """
        Parameters:
            x: base64 url data, or (if tool == "sketch") a dict of image and mask base64 url data
        Returns:
            image in requested format, or (if tool == "sketch") a dict of image and mask in requested format
        """
        if x is None:
            return x

        mask = ""
        if self.tool == "sketch" and self.source in ["upload", "webcam"]:
            assert isinstance(x, dict)
            x, mask = x["image"], x["mask"]

        assert isinstance(x, str) 
        if self.type == "filepath":
            content = x.partition(";")[2]
            image_encoded = content.partition(",")[2]
            fmt = x.partition(";")[0].partition("image/")[2]
            file_obj = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=("." + fmt.lower() if fmt is not None else ".png"),
            )
            open(file_obj.name, "wb").write(base64.b64decode(image_encoded))
            return file_obj.name
        
        im = processing_utils.decode_base64_to_image(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = im.convert(self.image_mode)
        if self.shape is not None:
            im = processing_utils.resize_and_crop(im, self.shape)
        if self.invert_colors:
            im = PIL.ImageOps.invert(im)
        if (
            self.source == "webcam"
            and self.mirror_webcam is True
            and self.tool != "color-sketch"
        ):
            im = PIL.ImageOps.mirror(im)

        if self.tool == "sketch" and self.source in ["upload", "webcam"]:
            mask_im = processing_utils.decode_base64_to_image(mask)
            return {
                "image": self._format_image(im),
                "mask": self._format_image(mask_im),
            }

        return self._format_image(im)

    def postprocess(
        self, y: np.ndarray | _Image.Image | str | Path | None
    ) -> str | None:
        """
        Parameters:
            y: image as a numpy array, PIL Image, string/Path filepath, or string URL
        Returns:
            base64 url data
        """
        if y is None:
            return None
        if isinstance(y, np.ndarray):
            return processing_utils.encode_array_to_base64(y)
        elif isinstance(y, _Image.Image):
            return processing_utils.encode_pil_to_base64(y)
        elif isinstance(y, (str, Path)):
            return processing_utils.encode_url_or_file_to_base64(y)
        else:
            raise ValueError("Cannot process this value as an Image")

    def set_interpret_parameters(self, segments: int = 16):
        """
        Calculates interpretation score of image subsections by splitting the image into subsections, then using a "leave one out" method to calculate the score of each subsection by whiting out the subsection and measuring the delta of the output value.
        Parameters:
            segments: Number of interpretation segments to split image into.
        """
        self.interpretation_segments = segments
        return self

    def _segment_by_slic(self, x):
        """
        Helper method that segments an image into superpixels using slic.
        Parameters:
            x: base64 representation of an image
        """
        x = processing_utils.decode_base64_to_image(x)
        if self.shape is not None:
            x = processing_utils.resize_and_crop(x, self.shape)
        resized_and_cropped_image = np.array(x)
        try:
            from skimage.segmentation import slic
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Error: running this interpretation for images requires scikit-image, please install it first."
            )
        try:
            segments_slic = slic(
                resized_and_cropped_image,
                self.interpretation_segments,
                compactness=10,
                sigma=1,
                start_label=1,
            )
        except TypeError:  # For skimage 0.16 and older
            segments_slic = slic(
                resized_and_cropped_image,
                self.interpretation_segments,
                compactness=10,
                sigma=1,
            )
        return segments_slic, resized_and_cropped_image

    def tokenize(self, x):
        """
        Segments image into tokens, masks, and leave-one-out-tokens
        Parameters:
            x: base64 representation of an image
        Returns:
            tokens: list of tokens, used by the get_masked_input() method
            leave_one_out_tokens: list of left-out tokens, used by the get_interpretation_neighbors() method
            masks: list of masks, used by the get_interpretation_neighbors() method
        """
        segments_slic, resized_and_cropped_image = self._segment_by_slic(x)
        tokens, masks, leave_one_out_tokens = [], [], []
        replace_color = np.mean(resized_and_cropped_image, axis=(0, 1))
        for (i, segment_value) in enumerate(np.unique(segments_slic)):
            mask = segments_slic == segment_value
            image_screen = np.copy(resized_and_cropped_image)
            image_screen[segments_slic == segment_value] = replace_color
            leave_one_out_tokens.append(
                processing_utils.encode_array_to_base64(image_screen)
            )
            token = np.copy(resized_and_cropped_image)
            token[segments_slic != segment_value] = 0
            tokens.append(token)
            masks.append(mask)
        return tokens, leave_one_out_tokens, masks

    def get_masked_inputs(self, tokens, binary_mask_matrix):
        masked_inputs = []
        for binary_mask_vector in binary_mask_matrix:
            masked_input = np.zeros_like(tokens[0], dtype=int)
            for token, b in zip(tokens, binary_mask_vector):
                masked_input = masked_input + token * int(b)
            masked_inputs.append(processing_utils.encode_array_to_base64(masked_input))
        return masked_inputs

    def get_interpretation_scores(
        self, x, neighbors, scores, masks, tokens=None, **kwargs
    ) -> List[List[float]]:
        """
        Returns:
            A 2D array representing the interpretation score of each pixel of the image.
        """
        x = processing_utils.decode_base64_to_image(x)
        if self.shape is not None:
            x = processing_utils.resize_and_crop(x, self.shape)
        x = np.array(x)
        output_scores = np.zeros((x.shape[0], x.shape[1]))

        for score, mask in zip(scores, masks):
            output_scores += score * mask

        max_val, min_val = np.max(output_scores), np.min(output_scores)
        if max_val > 0:
            output_scores = (output_scores - min_val) / (max_val - min_val)
        return output_scores.tolist()

    def style(self, *, height: int | None = None, width: int | None = None, **kwargs):
        """
        This method can be used to change the appearance of the Image component.
        Parameters:
            height: Height of the image.
            width: Width of the image.
        """
        self._style["height"] = height
        self._style["width"] = width
        return Component.style(
            self,
            **kwargs,
        )

    def stream(
        self,
        fn: Callable,
        inputs: List[Component],
        outputs: List[Component],
        _js: str | None = None,
        api_name: str | None = None,
        preprocess: bool = True,
        postprocess: bool = True,
    ):
        """
        This event is triggered when the user streams the component (e.g. a live webcam
        component)
        Parameters:
            fn: Callable function
            inputs: List of inputs
            outputs: List of outputs
        """
        # js: Optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
        if self.source != "webcam":
            raise ValueError("Image streaming only available if source is 'webcam'.")
        super().stream(
            fn,
            inputs,
            outputs,
            _js=_js,
            api_name=api_name,
            preprocess=preprocess,
            postprocess=postprocess,
        )

    def as_example(self, input_data: str | None) -> str:
        if input_data is None:
            return ""
        elif (
            self.root_url
        ):  # If an externally hosted image, don't convert to absolute path
            return input_data
        return str(utils.abspath(input_data))

