import numpy as np
from PIL import Image
from scipy.optimize import minimize

from .purifier import PurifierBase


class TVM(PurifierBase):
    def __init__(self):
        TVM.ARGS = {'prob': 0.3, 'norm': 2, 'lamb': 0.5,
                    'solver': 'L-BFGS-B', 'max_iter': 10,
                    'clip_values': None}
        """
        Create an instance of total variance minimization.

        :param prob: Probability of the Bernoulli distribution.
        :param norm: The norm (positive integer).
        :param lamb: The lambda parameter in the objective function.
        :param solver: Current support: `L-BFGS-B`, `CG`, `Newton-CG`.
        :param max_iter: Maximum number of iterations when performing optimization.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        """

    def purify(self, model, x, x_trans, *args):
        args = self._parameter_check(args)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        x_numpy = np.array(x.copy())
        z_min = x_numpy.copy()

        mask = (np.random.rand(*x_numpy.shape) < args['prob']).astype("int")
        for i in range(x_numpy.shape[2]):
            res = minimize(
                self._loss_func,
                z_min[:, :, i].flatten(),
                (x_numpy[:, :, i], mask[:, :, i], args['norm'], args['lamb']),
                method=args['solver'],
                jac=self._deri_loss_func,
                options={"maxiter": args['max_iter']},
            )
            z_min[:, :, i] = np.reshape(res.x, z_min[:, :, i].shape)

        if args['clip_values'] is not None:
            z_min = z_min.clamp(args['clip_values'][0], args['clip_values'][1])

        return Image.fromarray(z_min)

    @staticmethod
    def _loss_func(z_init, x, mask, norm, lamb):
        '''Regular gradient descent. Pass the function to be minimised, its derivative and the initial value'''

        res = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        z_init = np.reshape(z_init, x.shape)
        res += lamb * np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1).sum()
        res += lamb * np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0).sum()

        return res

    @staticmethod
    def _deri_loss_func(z_init, x, mask, norm, lamb):
        nor1 = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        nor1 = max(nor1, 1e-06)
        der1 = ((z_init - x.flatten()) * mask.flatten()) / (nor1 * 1.0)

        z_init = np.reshape(z_init, x.shape)

        if norm == 1:
            z_d1 = np.sign(z_init[1:, :] - z_init[:-1, :])
            z_d2 = np.sign(z_init[:, 1:] - z_init[:, :-1])
        else:
            z_d1_norm = np.power(np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1), norm - 1)
            z_d2_norm = np.power(np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0), norm - 1)
            z_d1_norm[z_d1_norm < 1e-6] = 1e-6
            z_d2_norm[z_d2_norm < 1e-6] = 1e-6
            z_d1_norm = np.repeat(z_d1_norm[:, np.newaxis], z_init.shape[1], axis=1)
            z_d2_norm = np.repeat(z_d2_norm[np.newaxis, :], z_init.shape[0], axis=0)
            z_d1 = norm * np.power(z_init[1:, :] - z_init[:-1, :], norm - 1) / z_d1_norm
            z_d2 = norm * np.power(z_init[:, 1:] - z_init[:, :-1], norm - 1) / z_d2_norm

        der2 = np.zeros(z_init.shape)
        der2[:-1, :] -= z_d1
        der2[1:, :] += z_d1
        der2[:, :-1] -= z_d2
        der2[:, 1:] += z_d2
        der2 = lamb * der2.flatten()

        return der1 + der2
