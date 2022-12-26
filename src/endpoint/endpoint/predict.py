from diffusers import StableDiffusionInpaintPipeline
import torch
from typing import Union
from .logger import get_logger

logger = get_logger(__name__)


class Model:
    def __init__(self):
        self.load_model()

    def predict(self, *args, **kwargs) -> Union[dict, tuple]:
        """Runs prediction on GPU if is available.

            Args:
                See `StableDiffusionInpaintPipeline` in `diffusers` for args and kwargs.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple. (default true) type: ordered dict
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        predictions = self.model(*args, **kwargs)
        logger.info("predictions created")
        return predictions

    def to_gpu(
        self, model: StableDiffusionInpaintPipeline
    ) -> StableDiffusionInpaintPipeline:
        """Casts torch model to gpu if available."""
        device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        model.to(device)
        logger.info("model casted to %s" % device_type)
        return model

    def load_model(self, model_dir: str) -> StableDiffusionInpaintPipeline:
        """Load the saved model from the s3 bucket for the StableDiffuserInpaintPipeline (SDIP).

        Pipeline for text-guided image inpainting using Stable Diffusion.
        This model inherits from [`DiffusionPipeline`].

        Args:
            model_dir (str): local directory to which to save the model
        """
        model = StableDiffusionInpaintPipeline.from_pretrained(
            model_dir,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        logger.info("model loaded")
        self.model = self.to_gpu(model)
