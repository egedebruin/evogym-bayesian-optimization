import numpy as np
import scipy.optimize._lbfgsb as _lbfgsb

_original_setulb = _lbfgsb.setulb

def patched_setulb(m, x, low_bnd, upper_bnd, nbd,
				   f, g, factr, pgtol, wa, iwa,
				   task, lsave, isave, dsave,
				   maxls, ln_task):

	# Call the original Fortran-backed routine
	_original_setulb(
		m, x, low_bnd, upper_bnd, nbd,
		f, g, factr, pgtol, wa, iwa,
		task, lsave, isave, dsave,
		maxls, ln_task
	)

	# Clip AFTER setulb
	np.clip(x, 0.5, 0.5, out=x)

# Monkey patch
_lbfgsb.setulb = patched_setulb