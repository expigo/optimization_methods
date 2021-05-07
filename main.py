from dmdo import dmdo

#dmdo(ts=[1e-2, 15e-2, 20e-2, 25e-2, 30e-2, 35e-2, 45e-2], K=100, opt_t=True, cg=True)
# dmdo(ts=[20e-2, 25e-2, 30e-2, 35e-2], K=100, opt_t=True, cg=False)
#dmdo(ts=[35e-2], K=200, opt_t=False, cg=True)
dmdo(ts=[30e-2], K=100, opt_t=False, cg=True, multimode=True)
