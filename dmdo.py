import numpy as np
from sympy import symbols, diff
from matplotlib import pyplot as plt
from scipy.optimize import fmin, minimize



# plt.xkcd()
# plt.style.use("fivethirtyeight")
plt.style.use("ggplot")


def dmdo(cg=False, opt_t=False, ts=[45e-2], K=25, multimode=False):
    # Min <- J = 1/2 E(qxi^2+rui^2) + (1/2)fxN^2
    N = 5
    a = 1
    b = 1/2
    q = 1
    r = .5
    f = 1.5

    n_iter = K
    epsilon = 15e-2

    X = np.zeros(shape=(n_iter, N + 1))
    P = np.zeros(shape=(n_iter, N))
    B = np.zeros(shape=(n_iter, 1))
    U = np.zeros(shape=(n_iter, N))
    J = np.zeros(shape=(len(ts), n_iter))
    d = np.zeros(shape=(n_iter, N))

    # u = np.random.uniform(low=0, high=20, size=N)  # 1st control randomization
    x_0 = 2

    #use_CG = cg        # use conjugate method
    #optimize_t = opt_t    # use optimized step value

    def _optimize(t, use_CG=cg, optimize_t=opt_t, epsilon=epsilon):
        nonlocal u
        for i in range(n_iter):

            x = np.insert(x_0 + np.cumsum(u) / 2, 0, x_0)  # calculate controls
            X[i, :] = x  # append xi to array, just to have a summary of changes

            p_N = 3 * x[-1] / 2  # calculate the value of the Nth costate (since dx_N != 0)
            p_reversed = p_N + np.cumsum(x[-2::-1])  # calc the adjoint vector
            P[i, :] = p_reversed  # keep them for later

            b = np.add(u, np.insert(p_reversed, 0, p_N)[-2::-1]) / 2  # calculate gradients
            norm = np.linalg.norm(b)  # compute gradient norm
            B[i] = norm  # store for convenience

            if norm < epsilon:  # stop condition
                print(f'Optimal solution found after {i} iterations '
                      f'(method: {"Direct" if not use_CG else "Conjugated"} Gradient | epsilon={epsilon} | norm={norm} | t={t if not optimize_t else "optimal"})')
                J[idx, i:] = J[idx,i-1]
                break

            additional_comp = (1 / 2) * (3 / 2) * x[-1] ** 2
            J_curr = np.sum(x[:-1] ** 2 + (1/2) * u ** 2) / 2 + (1 / 2) * (3 / 2) * x[-1] ** 2
            J[idx, i] = J_curr

            if optimize_t:
                # J_t = lambda t: (np.sum((x[:-1] + (u-t*b)/2)**2 + ((u-t*b)**2)/2))/2 +\
                #                 (3/4) * ((x_0 + np.cumsum(u-t*b)/2)[-1])**2

                J_t = lambda t: (np.sum(
                    (np.insert(x_0 + np.cumsum(u) / 2, 0, x_0))[:-1] ** 2 + ((u - t * b) ** 2) / 2)) / 2 + \
                                (3 / 4) * ((x_0 + np.cumsum(u - t * b) / 2)[-1]) ** 2

                def Jt(t):
                    u_next = u - t * b
                    x_next = np.insert(x_0 + np.cumsum(u_next) / 2, 0, x_0)
                    return (1 / 2) * np.sum(x_next[:-1] ** 2 + (1 / 2) * u_next ** 2) + (3 / 4) * x_next[-1] ** 2

                t = minimize(fun=Jt, method="Nelder-Mead", x0=0).x
                #if t == 0:
                #    print(f'Optimal solution found after {i} iterations (epsilon={epsilon} | norm={norm})')

                # print(i, t, B[i])

            if use_CG:
                if i == 0:
                    u = u - t * b
                    U[i, :] = u
                    d[i, :] = -b
                else:
                    c = B[i] ** 2 / B[i - 1] ** 2
                    d[i, :] = -b + c * d[i - 1, :]
                    u = u + t * d[i, :]
                    U[i, :] = u
            else:
                u = u - t * b
                U[i, :] = u

        #plt.plot(J[idx, :], label=f't={t}')

    if multimode:
        J = np.zeros(shape=(4, n_iter))

        idx=0
        u = np.array([2, 16, 2, 8, 4])  # same as in the initial test
        _optimize(epsilon=1e-5, t=ts[0], use_CG=False, optimize_t=False)
        plt.plot(J[idx, :], label=f't={ts}')

        idx=1
        u = np.array([2, 16, 2, 8, 4])  # same as in the initial test
        _optimize(epsilon=1e-5, t=ts[0], use_CG=True, optimize_t=False)
        plt.plot(J[idx, :], label=f't={ts} [Conjugate]')

        idx=2
        u = np.array([2, 16, 2, 8, 4])  # same as in the initial test
        _optimize(epsilon=1e-5, t=ts[0], use_CG=False, optimize_t=True)
        plt.plot(J[idx, :], label=f't-optimized')

        idx=3
        u = np.array([2, 16, 2, 8, 4])  # same as in the initial test
        _optimize(epsilon=1e-5, t=ts[0], use_CG=True, optimize_t=True)
        plt.plot(J[idx, :], label=f't-optimized [Conjugate]')

        plt.title("Mix")
        plt.xlabel("Iteration")
        plt.ylabel("Performance Index")
        xlim = 20
        ylim = 40
        plt.xticks(np.arange(0, xlim, step=1))
        plt.yticks(np.arange(0, ylim, step=2))
        plt.xlim((0, xlim))
        plt.ylim((0, ylim))

        plt.legend()
        plt.show()

    else:
        for idx, t in enumerate(ts):
            u = np.array([2, 16, 2, 8, 4])  # same as in the initial test
            if opt_t:
                _optimize(t, optimize_t=True)
                plt.plot(J[idx, :], label=f't-optimal')
                xlim = 20
                ylim = 50
                plt.xticks(np.arange(0, xlim, step=1))
                plt.yticks(np.arange(0, ylim, step=2))
                plt.xlim((0, xlim))
                plt.ylim((0, ylim))
                break

            else:
                _optimize(t, optimize_t=False)
                plt.plot(J[idx, :], label=f't={t}')
                xlim, dx = (20, 1) if cg else (30, 1)
                ylim = 50
                plt.xticks(np.arange(0, xlim, step=dx))
                plt.yticks(np.arange(0, ylim, step=2))
                plt.xlim((0, xlim))
                plt.ylim((0, ylim))

            # plt.plot(J[idx, :], label=f't={t}')

        # plt.tight_layout()
        plt.title(f'{"Conjugated" if cg else "Direct"} Gradient')
        plt.xlabel("Iteration")
        plt.ylabel("Performance Index")
        plt.legend()
        plt.show()
    print()
