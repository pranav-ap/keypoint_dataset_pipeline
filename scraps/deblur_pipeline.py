from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from skimage.util import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm


@dataclass
class InferenceConfig:
    data_path = "assets/"
    image_data_path = "/kaggle/working/"


config = InferenceConfig()


class DeblurDatasetPipeline:
    def __init__(self):
        parameters = {
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [4, 6, 6, 8],
            'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.66,
            'bias': False,
            'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False
        }

        self.task = "Single_Image_Defocus_Deblurring"
        weights, parameters = self.get_weights_and_parameters(self.task, parameters)

        from runpy import run_path
        load_arch = run_path(os.path.join('Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))

        self.model = load_arch['Restormer'](**parameters)
        self.model.cuda()

        checkpoint = torch.load(weights)
        self.model.load_state_dict(checkpoint['params'])
        self.model.eval()

    @staticmethod
    def get_weights_and_parameters(task, parameters):
        weights = None

        if task == 'Motion_Deblurring':
            weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
        elif task == 'Single_Image_Defocus_Deblurring':
            weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
        elif task == 'Real_Denoising':
            weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
            parameters['LayerNorm_type'] = 'BiasFree'

        return weights, parameters

    """
    DEBLUR
    """

    def deblur(self, filepath):
        img_multiple_of = 8

        with (torch.no_grad()):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
            input_ = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).cuda()

            # Pad the input if not multiple of 8
            h = input_.shape[2]
            w = input_.shape[3]

            H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
            W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of

            padh = H - h if h % img_multiple_of != 0 else 0
            padw = W - w if w % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = self.model(input_)
            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :h, :w]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            return restored

    """
    RUN
    """

    def run(self):
        input_dir = 'demo/sample_images/' + self.task + '/degraded'
        os.makedirs(input_dir, exist_ok=True)
        out_dir = 'demo/sample_images/' + self.task + '/restored'
        os.makedirs(out_dir, exist_ok=True)

        files = natsorted(glob(os.path.join(input_dir, '*')))

        for filepath in tqdm(files):
            restored = self.deblur(filepath)

            filename = os.path.split(filepath)[-1]

            cv2.imwrite(
                os.path.join(out_dir, filename),
                cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
            )


def main():
    pipeline = DeblurDatasetPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
