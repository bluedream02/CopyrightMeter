from .purifier import PurifierBase


class Quantize(PurifierBase):
    def __init__(self):
        Quantize.ARGS = {'bits': 8}

    def purify(self, model, x, x_trans, *args):
        args = self._parameter_check(args)

        return x.quantize(colors=2**args['bits'])