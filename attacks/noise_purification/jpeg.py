import io
import torchvision.transforms as T
from PIL import Image
from .purifier import PurifierBase


class JPEG(PurifierBase):
    def __init__(self):
        JPEG.ARGS = {'quality': 75}

    def purify(self, model, x, x_trans, *args):
        args = self._parameter_check(args)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        output_stream = io.BytesIO()

        x.save(output_stream, format='JPEG', quality=args['quality'])
        compressed_image_data = output_stream.getvalue()
        output_stream.close()

        return Image.open(io.BytesIO(compressed_image_data))
