class PurifierBase(object):
    """
    The base class of all attackers.
    """

    ARGS = dict()

    def __call__(self, model):
        raise NotImplementedError()

    def _parameter_check(self, *args):
        parameter = dict()
        for i in self.ARGS:
            if i in args:
                parameter[i] = args[i]
            else:
                parameter[i] = self.ARGS[i]
        return parameter

    def generate(self, model, input_, AttackGoal):
        raise NotImplementedError()
