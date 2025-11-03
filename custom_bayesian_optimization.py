from __future__ import annotations
from bayes_opt import BayesianOptimization

from configs import config


class CustomBayesianOptimization(BayesianOptimization):

    def suggest(self) -> dict[str, float]:
        """Suggest a promising point to probe next."""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Finding argmax of the acquisition function.
        suggestion = self._acquisition_function.suggest(gp=self._gp, target_space=self._space, fit_gp=True, n_l_bfgs_b=config.BO_RESTARTS)

        return self._space.array_to_params(suggestion)
