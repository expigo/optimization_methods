import argparse
from dmdo import dmdo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Optimization without Constraints")

    parser.add_argument("-c", "--conjugated-gradient", action="store_true", help="Use Conjugated Gradient")
    parser.add_argument("-t0", "--optimize-step", action="store_true", help="Optimize the step value")
    parser.add_argument("-K", "--no_iterations", default=50, type=int, help="Number of iterations (default: 50)")
    parser.add_argument("-e", "-epsilon", default=0.01, help="Desired accuracy")
    parser.add_argument("-ts", "-steps", nargs="*", default=[0.35], type=float, help="constant values of steps to test")
    parser.add_argument("-a", "-all", action="store_true", help="Show comparison of all methods")
    args = parser.parse_args()
    print(args)

    dmdo(e=args.e, ts=args.ts, K=args.no_iterations, opt_t=args.optimize_step, cg=args.conjugated_gradient,
         multimode=args.a)


#dmdo(ts=[1e-2, 15e-2, 20e-2, 25e-2, 30e-2, 35e-2, 45e-2], K=100, opt_t=True, cg=True)
# dmdo(ts=[20e-2, 25e-2, 30e-2, 35e-2], K=100, opt_t=True, cg=False)
#dmdo(ts=[35e-2], K=200, opt_t=False, cg=True)
#dmdo(ts=[30e-2], K=100, opt_t=False, cg=True, multimode=True)nums = np.array([0, 1, 2, 3])

