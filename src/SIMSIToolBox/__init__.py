import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 500
import skimage.filters
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
from sklearn.manifold import TSNE
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from pyimzml.ImzMLWriter import ImzMLWriter
from multiprocessing import Pool,Manager
from PIL import Image
import picor
import molmass
from threading import Thread



def startConcurrentTask(task,args,numCores,message,total,chunksize="none",verbose=True):
    if verbose:
        m = Manager()
        q = m.Queue()
        args = [a + [q] for a in args]
        t = Thread(target=updateProgress, args=(q, total, message))
        t.start()
    if numCores > 1:
        p = Pool(numCores)
        if chunksize == "none":
            res = p.starmap(task, args)
        else:
            res = p.starmap(task, args, chunksize=chunksize)
        p.close()
        p.join()
    else:
        res = [task(*a) for a in args]
    if verbose: t.join()
    return res

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def updateProgress(q, total,message = ""):
    counter = 0
    while counter != total:
        if not q.empty():
            q.get()
            counter += 1
            printProgressBar(counter, total,prefix = message, printEnd="")

def objectiveFunc(t, p, goodInd, params=[], alpha=0, lam=0):
    trel = np.array([t[x] for x in goodInd])
    prel = np.array([p[x] for x in goodInd])

    trel = trel / np.sum(trel)
    prel = prel / np.sum(prel)

    return np.sum(np.square(np.subtract(trel, prel))) + alpha * lam * np.sum(np.abs(params)) + (
                1 - alpha) / 2 * lam * np.sum(np.square(params))

def ISAFit(T, N, P, func, goodInd, x_init=np.random.random((1)), plot=False,q=None):
    success = False
    initial_params = np.concatenate((x_init, T), axis=None)
    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, func(x[0], 1.0, [xx / np.sum(x[1:]) for xx in x[1:]], N, P), goodInd),
            x0=initial_params)  # ,method="trust-constr",
        # bounds=[(0, x) for x in [1,np.inf,np.inf,np.inf]])
        if not sol.success:
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    # g, D = sol.x[:2]
    g = sol.x[0]
    D = 1
    T = sol.x[1:]
    T = T / np.sum(T)
    err = sol.fun
    P_pred = func(g, D, T, N, P)
    P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0
    x_ind = 0
    x_lab = []
    maxY = np.max(np.concatenate((P, P_pred)))
    i = 0
    if plot:
        for p, pp in zip(P, P_pred):
            plt.bar([x_ind, x_ind + 1], [p, pp], color=["black", "red"])
            x_lab.append([x_ind + .5, "M+" + str(i)])
            x_ind += 4
            i += 1
        plt.xticks([x[0] for x in x_lab], [x[1] for x in x_lab], rotation=90)
        plt.scatter([-1], [-1], c="red", label="Predicted")
        plt.scatter([-1], [-1], c="black", label="Measured")
        plt.legend()
        plt.ylim((0, maxY))
        plt.xlim((-2, x_ind + 1))
        plt.figure()

        # plot solution curves
        D_test = np.linspace(0, 1, 25)
        for pp in range(len(P)):
            g_test = []
            for d in D_test:
                sol = opt.minimize(lambda x: abs(P[pp] - func(x[0], d, T, N, P)[pp]), x0=[g])
                g_test.append(sol.x[0])
            plt.plot(D_test, g_test, c="black")

        plt.scatter([D], [g], color="red")
        plt.ylim((0, 1))
        plt.xlabel("D")
        plt.ylabel("g(t)")

    if type(q) != type(None):
        q.put(0)

    return g, D, T, err, P_pred


def integratedISAFull(G, k1, k2, k3, k4, T, N, numC):
    func = getISAEq(numC)
    ts = np.linspace(0, 1, 20)
    vals = np.array([parameterizedIntegrandFull(t, N, G, k1, k2, k3, k4, T, func) for t in ts])
    output = np.array([np.trapz(vals[:, x], ts) for x in range(len(vals[0]))])
    return output


def parameterizedIntegrandFull(t, N, G, k1, k2, k3, k4, T, func):
    val = [1 - generalizedExp(t, 1 - T[0], k2), generalizedExp(t, T[1], k3), generalizedExp(t, T[2], k4)]
    val = val / np.sum(val)
    return np.array(func(full_dgdt(t, G, k1), 1.0, val, N, None))


def integrated_X_full(T, k2, k3, k4):
    val = [1 - generalizedExp(1, 1 - T[0], k2), generalizedExp(1, T[1], k3), generalizedExp(1, T[2], k4)]
    output = val / np.sum(val)

    return output


def integratedISA(G, k1, k2, T, N, numC):
    func = getISAEq(numC)
    ts = np.linspace(0, 1, 20)
    T = T / np.sum(T)
    vals = np.array([parameterizedIntegrand(t, N, G, k1, k2, T, func) for t in ts])
    output = np.array([np.trapz(vals[:, x], ts) for x in range(len(vals[0]))])
    return output

def generalizedExp(t, c, k):
    return c * (1 - np.exp(-1 * k * t))


def full_g_t(t, G, k):
    return generalizedExp(t, G, k)


def full_dgdt(t, G, k):
    return k * G * np.exp(-1 * k * t)


def d_t(t, D, k):
    return generalizedExp(t, D, k)


def parameterizedIntegrand(t, N, G, k1, k2, T, func):
    return np.array(func(full_dgdt(t, G, k1), d_t(t, 1, k2), T, N, None))


def integrated_X(T, N, k2):
    t = 1
    return d_t(t, 1, k2) * np.array(T) + (1 - d_t(t, 1, k2)) * np.array(N)

#
# def ISAFit_nonSS(T, N, P, numC, goodInd, x_init=np.random.random((3)), plot=False):
#     success = False
#
#     initial_params = np.concatenate((x_init, T), axis=None)
#     while not success:
#         sol = opt.minimize(
#             lambda x: objectiveFunc(P, integratedISA(x[0], x[1], x[2], [x[3], x[4], x[5]], N, numC), goodInd,
#                                     [x[0], x[4] / np.sum(x[3:]), x[5] / np.sum(x[3:])], alpha=0, lam=1e-2),
#             x0=initial_params,
#             bounds=[(0, m) for m in [1, np.inf, np.inf, np.inf, np.inf, np.inf]])
#         if not sol.success:
#             print("failed")
#             initial_params = np.random.random(initial_params.shape)
#         else:
#             success = True
#     # g, D = sol.x[:2]
#     g = full_g_t(1, sol.x[0], sol.x[1])
#     D = d_t(1, 1, sol.x[2])
#     T = sol.x[3:]
#     T = T / np.sum(T)
#
#     err = sol.fun
#     P_pred = integratedISA(sol.x[0], sol.x[1], sol.x[2], T, N, numC)
#     P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
#     for x in range(len(P_pred)):
#         if x not in goodInd:
#             P_pred[x] = 0
#     x_ind = 0
#     x_lab = []
#     maxY = np.max(np.concatenate((P, P_pred)))
#     i = 0
#     if plot:
#         for p, pp in zip(P, P_pred):
#             plt.bar([x_ind, x_ind + 1], [p, pp], color=["black", "red"])
#             x_lab.append([x_ind + .5, "M+" + str(i)])
#             x_ind += 4
#             i += 1
#         plt.xticks([x[0] for x in x_lab], [x[1] for x in x_lab], rotation=90)
#         plt.scatter([-1], [-1], c="red", label="Predicted")
#         plt.scatter([-1], [-1], c="black", label="Measured")
#         plt.legend()
#         plt.ylim((0, maxY))
#         plt.xlim((-2, x_ind + 1))
#
#     T = integrated_X(T, N, sol.x[2])
#     return g, D, T, err, P_pred
#
#
# def ISAFit_nonSS_full(T, N, P, numC, goodInd, x_init=np.random.random((5)), plot=False):
#     success = False
#
#     initial_params = np.concatenate((x_init, T), axis=None)
#     while not success:
#         sol = opt.minimize(
#             lambda x: objectiveFunc(P, integratedISAFull(x[0], x[1], x[2], x[3], x[4], x[5:], N, numC), goodInd),
#             x0=initial_params,
#             bounds=[(0, m) for m in [1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]])
#         if not sol.success:
#             print("failed")
#             initial_params = np.random.random(initial_params.shape)
#         else:
#             success = True
#     # g, D = sol.x[:2]
#     g = full_g_t(1, sol.x[0], sol.x[1])
#     D = 1.0
#     T = sol.x[5:]
#     # T[0] = 1-T[0]
#     T = T / np.sum(T)
#
#     err = sol.fun
#     P_pred = integratedISAFull(sol.x[0], sol.x[1], sol.x[2], sol.x[3], sol.x[4], sol.x[5:], N, numC)
#     P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
#     for x in range(len(P_pred)):
#         if x not in goodInd:
#             P_pred[x] = 0
#     x_ind = 0
#     x_lab = []
#     maxY = np.max(np.concatenate((P, P_pred)))
#     i = 0
#     if plot:
#         for p, pp in zip(P, P_pred):
#             plt.bar([x_ind, x_ind + 1], [p, pp], color=["black", "red"])
#             x_lab.append([x_ind + .5, "M+" + str(i)])
#             x_ind += 4
#             i += 1
#         plt.xticks([x[0] for x in x_lab], [x[1] for x in x_lab], rotation=90)
#         plt.scatter([-1], [-1], c="red", label="Predicted")
#         plt.scatter([-1], [-1], c="black", label="Measured")
#         plt.legend()
#         plt.ylim((0, maxY))
#         plt.xlim((-2, x_ind + 1))
#
#     T = integrated_X_full(T, sol.x[2], sol.x[3], sol.x[4])
#
#     return g, D, T, err, P_pred


def ISAFit_classical(T, N, P, func, goodInd, x_init=np.random.random((2)), plot=False,q=None):
    success = False
    initial_params = x_init
    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, func(x[0], x[1], T, N, P), goodInd),
            x0=initial_params,
        )  # bounds=[(0, 1) for _ in range(len(x_init) + len(T))])
        if not sol.success:
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    # g, D = sol.x[:2]
    g = sol.x[0]
    D = sol.x[1]

    err = sol.fun
    P_pred = func(g, D, T, N, P)
    P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0
    x_ind = 0
    x_lab = []
    maxY = np.max(np.concatenate((P, P_pred)))
    i = 0
    if plot:
        for p, pp in zip(P, P_pred):
            plt.bar([x_ind, x_ind + 1], [p, pp], color=["black", "red"])
            x_lab.append([x_ind + .5, "M+" + str(i)])
            x_ind += 4
            i += 1
        plt.xticks([x[0] for x in x_lab], [x[1] for x in x_lab], rotation=90)
        plt.scatter([-1], [-1], c="red", label="Predicted")
        plt.scatter([-1], [-1], c="black", label="Measured")
        plt.legend()
        plt.ylim((0, maxY))
        plt.xlim((-2, x_ind + 1))
        plt.figure()

        # plot solution curves
        D_test = np.linspace(0, 1, 25)
        for pp in range(len(P)):
            g_test = []
            for d in D_test:
                sol = opt.minimize(lambda x: abs(P[pp] - func(x[0], d, T, N, P)[pp]), x0=[g])
                g_test.append(sol.x[0])
            plt.plot(D_test, g_test, c="black")

        plt.scatter([D], [g], color="red")
        plt.ylim((0, 1))
        plt.xlabel("D")
        plt.ylabel("g(t)")

    if type(q) != type(None):
        q.put(0)

    return g, D, T, err, P_pred

def ISAFit_knownT(T, N, P, func, goodInd, x_init=np.random.random((1)), plot=False,q=None):
    sol = opt.minimize(
        lambda x: objectiveFunc(P, func(x[0], 1.0, T, N, P), goodInd),
        x0=x_init)
    g = sol.x[:1]
    D = 1.0
    err = sol.fun
    P_pred = func(g, D, T, N, P)
    P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0
    x_ind = 0
    x_lab = []
    i = 0

    if plot:
        maxY = np.max(np.concatenate((P, P_pred)))

        for p, pp in zip(P, P_pred):
            plt.bar([x_ind, x_ind + 1], [p, pp],color=["black","red"])
            x_lab.append([x_ind + .5, "M+" + str(i)])
            x_ind += 4
            i += 1
        plt.xticks([x[0] for x in x_lab], [x[1] for x in x_lab], rotation=90)
        plt.scatter([-1], [-1], c="red", label="Predicted")
        plt.scatter([-1], [-1], c="black", label="Measured")
        plt.legend()
        plt.ylim((0, maxY))
        plt.xlim((-2, x_ind + 1))

        plt.figure()

        # plot solution curves
        D_test = np.linspace(0, 1, 25)
        for pp in range(len(P)):
            g_test = []
            for d in D_test:
                sol = opt.minimize(lambda x: abs(P[pp] - func(x[0], d, T, N, P)[pp]), x0=[g])
                g_test.append(sol.x[0])
            plt.plot(D_test, g_test, c="black")

        plt.scatter([D], [g], color="red")
        plt.ylim((0, 1))
        plt.xlabel("D")
        plt.ylabel("g(t)")

    if type(q) != type(None):
        q.put(0)

    return g, D, T, err, P_pred


def convolveLayer(offset, height, width, layer, imageBoundary, method="MA",q=None):
    # iterate through pixels
    if method == "MA":
        tensorFilt = np.zeros((height - 2 * offset, width - 2 * offset))
        for r in range(offset, height - offset):
            for c in range(offset, width - offset):
                tempMat = layer[r - offset:r + offset + 1, c - offset:c + offset + 1]
                coef = imageBoundary[r - offset:r + offset + 1, c - offset:c + offset + 1]
                coef = coef / max([1, np.sum(coef)])
                tensorFilt[r - offset, c - offset] = np.sum(np.multiply(tempMat, coef))
    elif method == "GB":
        tensorFilt = ndimage.gaussian_filter(layer, offset)

    if type(q) != type(None):
        q.put(0)

    return tensorFilt


# imput matrix with half feature minimum
def imputeRowMin(arr, alt_min=2):
    # find the minimum non-zero value of each compound
    numImp = 0
    max_vals = []
    for c in arr.transpose():
        tmp = [x for x in c if x > alt_min]
        if len(tmp) > 0:
            val = np.min(tmp)
        else:
            val = alt_min * 2
        max_vals.append(val)
    # impute values

    data_imp = np.zeros(arr.shape)

    for c in range(arr.shape[1]):
        for r in range(len(arr)):
            if arr[r, c] > alt_min:
                if np.isinf(arr[r, c]) or np.isnan(arr[r, c]):
                    print("bad", arr[r, c])
                data_imp[r, c] = arr[r, c]
            else:
                data_imp[r, c] = max_vals[c] / 2
                numImp += 1
            if data_imp[r, c] < 1e-3:
                data_imp[r, c] = alt_min
                numImp += 1
    return data_imp





def write_file_to_zip(myzip, filename):
    myzip.write(filename,
                arcname=filename.split("/")[-1])


def myristicISA(g, D, T, N, P):
    # define tracer and naturual abundance isotopomers
    N0 = N[0]
    N1 = N[1]
    N2 = N[2]
    T0 = T[0]
    T1 = T[1]
    T2 = T[2]

    # compute X values
    X0 = D * T0 + (1 - D) * N0
    X1 = D * T1 + (1 - D) * N1
    X2 = D * T2 + (1 - D) * N2

    # compute product isotopmer abundances
    P = [g * (X0 ** 7) + (1 - g) * (N0 ** 7),
         g * (7 * (X0 ** 6) * X1) + (1 - g) * (7 * (N0 ** 6) * N1),
         g * (21 * (X0 ** 5) * (X1 ** 2) + 7 * (X0 ** 6) * X2) + (1 - g) * (
                 21 * (N0 ** 5) * (N1 ** 2) + 7 * (N0 ** 6) * N2),
         g * (35 * X0 ** 4 * X1 ** 3 + 42 * X0 ** 5 * X1 * X2) + (1 - g) * (
                 35 * N0 ** 4 * N1 ** 3 + 42 * N0 ** 5 * N1 * N2),
         g * (35 * X0 ** 3 * X1 ** 4 + 105 * X0 ** 4 * X1 ** 2 * X2 + 21 * X0 ** 5 * X2 ** 2) + (1 - g) * (
                 35 * N0 ** 3 * N1 ** 4 + 105 * N0 ** 4 * N1 ** 2 * N2 + 21 * N0 ** 5 * N2 ** 2),
         g * (21 * X0 ** 2 * X1 ** 5 + 140 * X0 ** 3 * X1 ** 3 * X2 + 105 * X0 ** 4 * X1 * X2 ** 2) + (1 - g) * (
                 21 * N0 ** 2 * N1 ** 5 + 140 * N0 ** 3 * N1 ** 3 * N2 + 105 * N0 ** 4 * N1 * N2 ** 2),
         g * (
                 7 * X0 * X1 ** 6 + 105 * X0 ** 2 * X1 ** 4 * X2 + 210 * X0 ** 3 * X1 ** 2 * X2 ** 2 + 35 * X0 ** 4 * X2 ** 3) + (
                 1 - g) * (
                 7 * N0 * N1 ** 6 + 105 * N0 ** 2 * N1 ** 4 * N2 + 210 * N0 ** 3 * N1 ** 2 * N2 ** 2 + 35 * N0 ** 4 * N2 ** 3),
         g * (X1 ** 7 + 42 * X0 * X1 ** 5 * X2 + 210 * X0 ** 2 * X1 ** 3 * X2 ** 2 + 140 * X0 ** 3 * X1 * X2 ** 3) + (
                 1 - g) * (
                 N1 ** 7 + 42 * N0 * N1 ** 5 * N2 + 210 * N0 ** 2 * N1 ** 3 * N2 ** 2 + 140 * N0 ** 3 * N1 * N2 ** 3),
         g * (
                 7 * X1 ** 6 * X2 + 105 * X0 * X1 ** 4 * X2 ** 2 + 210 * X0 ** 2 * X1 ** 2 * X2 ** 3 + 35 * X0 ** 3 * X2 ** 4) + (
                 1 - g) * (
                 7 * N1 ** 6 * N2 + 105 * N0 * N1 ** 4 * N2 ** 2 + 210 * N0 ** 2 * N1 ** 2 * N2 ** 3 + 35 * N0 ** 3 * N2 ** 4),
         g * (21 * X1 ** 5 * X2 ** 2 + 140 * X0 * X1 ** 3 * X2 ** 3 + 105 * X0 ** 2 * X1 * X2 ** 4) + (1 - g) * (
                 21 * N1 ** 5 * N2 ** 2 + 140 * N0 * N1 ** 3 * N2 ** 3 + 105 * N0 ** 2 * N1 * N2 ** 4),
         g * (35 * X1 ** 4 * X2 ** 3 + 105 * X0 * X1 ** 2 * X2 ** 4 + 21 * X0 ** 2 * X2 ** 5) + (1 - g) * (
                 35 * N1 ** 4 * N2 ** 3 + 105 * N0 * N1 ** 2 * N2 ** 4 + 21 * N0 ** 2 * N2 ** 5),
         g * (35 * (X1 ** 3) * (X2 ** 4) + 42 * X0 * X1 * (X2 ** 5)) + (1 - g) * (
                 35 * (N1 ** 3) * (N2 ** 4) + 42 * N0 * N1 * N2 ** 5),
         g * (21 * X1 ** 2 * (X2 ** 5) + 7 * X0 * (X2 ** 6)) + (1 - g) * (
                 21 * (N1 ** 2) * (N2 ** 5) + 7 * N0 * (N2 ** 6)),
         g * (7 * X1 * (X2 ** 6)) + (1 - g) * (7 * N1 * (N2 ** 6)),
         g * (X2 ** 7) + (1 - g) * (N2 ** 7)]

    return P


def stearicISA(g, D, T, N, P):
    # define tracer and naturual abundance isotopomers
    N0 = N[0]
    N1 = N[1]
    N2 = N[2]
    T0 = T[0]
    T1 = T[1]
    T2 = T[2]

    # compute X values
    X0 = D * T0 + (1 - D) * N0
    X1 = D * T1 + (1 - D) * N1
    X2 = D * T2 + (1 - D) * N2

    # compute product isotopmer abundances

    P = [g * (X0 ** 9) + (1 - g) * (N0 ** 9),
         g * (9 * X0 ** 8 * X1) + (1 - g) * (9 * N0 ** 8 * N1),
         g * (9 * X0 ** 8 * X2 + 36 * X0 ** 7 * X1 ** 2) + (1 - g) * (9 * N0 ** 8 * N2 + 36 * N0 ** 7 * N1 ** 2),
         g * (72 * X0 ** 7 * X1 * X2 + 84 * X0 ** 6 * X1 ** 3) + (1 - g) * (
                 72 * N0 ** 7 * N1 * N2 + 84 * N0 ** 6 * N1 ** 3),
         g * (36 * X0 ** 7 * X2 ** 2 + 252 * X0 ** 6 * X1 ** 2 * X2 + 126 * X0 ** 5 * X1 ** 4) + (1 - g) * (
                 36 * N0 ** 7 * N2 ** 2 + 252 * N0 ** 6 * N1 ** 2 * N2 + 126 * N0 ** 5 * N1 ** 4),
         g * (252 * X0 ** 6 * X1 * X2 ** 2 + 504 * X0 ** 5 * X1 ** 3 * X2 + 126 * X0 ** 4 * X1 ** 5) + (1 - g) * (
                 252 * N0 ** 6 * N1 * N2 ** 2 + 504 * N0 ** 5 * N1 ** 3 * N2 + 126 * N0 ** 4 * N1 ** 5),
         g * (
                 84 * X0 ** 6 * X2 ** 3 + 756 * X0 ** 5 * X1 ** 2 * X2 ** 2 + 630 * X0 ** 4 * X1 ** 4 * X2 + 84 * X0 ** 3 * X1 ** 6) + (
                 1 - g) * (
                 84 * N0 ** 6 * N2 ** 3 + 756 * N0 ** 5 * N1 ** 2 * N2 ** 2 + 630 * N0 ** 4 * N1 ** 4 * N2 + 84 * N0 ** 3 * N1 ** 6),
         g * (
                 504 * X0 ** 5 * X1 * X2 ** 3 + 1260 * X0 ** 4 * X1 ** 3 * X2 ** 2 + 504 * X0 ** 3 * X1 ** 5 * X2 + 36 * X0 ** 2 * X1 ** 7) + (
                 1 - g) * (
                 504 * N0 ** 5 * N1 * N2 ** 3 + 1260 * N0 ** 4 * N1 ** 3 * N2 ** 2 + 504 * N0 ** 3 * N1 ** 5 * N2 + 36 * N0 ** 2 * N1 ** 7),
         g * (
                 126 * X0 ** 5 * X2 ** 4 + 1260 * X0 ** 4 * X1 ** 2 * X2 ** 3 + 1260 * X0 ** 3 * X1 ** 4 * X2 ** 2 + 252 * X0 ** 2 * X1 ** 6 * X2 + 9 * X0 * X1 ** 8) + (
                 1 - g) * (
                 126 * N0 ** 5 * N2 ** 4 + 1260 * N0 ** 4 * N1 ** 2 * N2 ** 3 + 1260 * N0 ** 3 * N1 ** 4 * N2 ** 2 + 252 * N0 ** 2 * N1 ** 6 * N2 + 9 * N0 * N1 ** 8),
         g * (
                 630 * X0 ** 4 * X1 * X2 ** 4 + 1680 * X0 ** 3 * X1 ** 3 * X2 ** 3 + 756 * X0 ** 2 * X1 ** 5 * X2 ** 2 + 72 * X0 * X1 ** 7 * X2 + X1 ** 9) + (
                 1 - g) * (
                 630 * N0 ** 4 * N1 * N2 ** 4 + 1680 * N0 ** 3 * N1 ** 3 * N2 ** 3 + 756 * N0 ** 2 * N1 ** 5 * N2 ** 2 + 72 * N0 * N1 ** 7 * N2 + N1 ** 9),
         g * (
                 126 * X0 ** 4 * X2 ** 5 + 1260 * X0 ** 3 * X1 ** 2 * X2 ** 4 + 1260 * X0 ** 2 * X1 ** 4 * X2 ** 3 + 252 * X0 * X1 ** 6 * X2 ** 2 + 9 * X1 ** 8 * X2) + (
                 1 - g) * (
                 126 * N0 ** 4 * N2 ** 5 + 1260 * N0 ** 3 * N1 ** 2 * N2 ** 4 + 1260 * N0 ** 2 * N1 ** 4 * N2 ** 3 + 252 * N0 * N1 ** 6 * N2 ** 2 + 9 * N1 ** 8 * N2),
         g * (
                 504 * X0 ** 3 * X1 * X2 ** 5 + 1260 * X0 ** 2 * X1 ** 3 * X2 ** 4 + 504 * X0 * X1 ** 5 * X2 ** 3 + 36 * X1 ** 7 * X2 ** 2) + (
                 1 - g) * (
                 504 * N0 ** 3 * N1 * N2 ** 5 + 1260 * N0 ** 2 * N1 ** 3 * N2 ** 4 + 504 * N0 * N1 ** 5 * N2 ** 3 + 36 * N1 ** 7 * N2 ** 2),
         g * (
                 84 * X0 ** 3 * X2 ** 6 + 756 * X0 ** 2 * X1 ** 2 * X2 ** 5 + 630 * X0 * X1 ** 4 * X2 ** 4 + 84 * X1 ** 6 * X2 ** 3) + (
                 1 - g) * (
                 84 * N0 ** 3 * N2 ** 6 + 756 * N0 ** 2 * N1 ** 2 * N2 ** 5 + 630 * N0 * N1 ** 4 * N2 ** 4 + 84 * N1 ** 6 * N2 ** 3),
         g * (252 * X0 ** 2 * X1 * X2 ** 6 + 504 * X0 * X1 ** 3 * X2 ** 5 + 126 * X1 ** 5 * X2 ** 4) + (1 - g) * (
                 252 * N0 ** 2 * N1 * N2 ** 6 + 504 * N0 * N1 ** 3 * N2 ** 5 + 126 * N1 ** 5 * N2 ** 4),
         g * (36 * X0 ** 2 * X2 ** 7 + 252 * X0 * X1 ** 2 * X2 ** 6 + 126 * X1 ** 4 * X2 ** 5) + (1 - g) * (
                 36 * N0 ** 2 * N2 ** 7 + 252 * N0 * N1 ** 2 * N2 ** 6 + 126 * N1 ** 4 * N2 ** 5),
         g * (72 * X0 * X1 * X2 ** 7 + 84 * X1 ** 3 * X2 ** 6) + (1 - g) * (
                 72 * N0 * N1 * N2 ** 7 + 84 * N1 ** 3 * N2 ** 6),
         g * (9 * X0 * X2 ** 8 + 36 * X1 ** 2 * X2 ** 7) + (1 - g) * (9 * N0 * N2 ** 8 + 36 * N1 ** 2 * N2 ** 7),
         g * (9 * X1 * X2 ** 8) + (1 - g) * (9 * N1 * N2 ** 8),
         g * (X2 ** 9) + (1 - g) * (N2 ** 9)]

    return P


def palmitateISA(g, D, T, N, P):
    # define tracer and naturual abundance isotopomers
    N0 = N[0]
    N1 = N[1]
    N2 = N[2]
    T0 = T[0]
    T1 = T[1]
    T2 = T[2]

    # compute X values
    X0 = D * T0 + (1 - D) * N0
    X1 = D * T1 + (1 - D) * N1
    X2 = D * T2 + (1 - D) * N2

    # compute product isotopmer abundances

    P = [g * (X0 ** 8) + (1 - g) * (N0 ** 8),
         g * (8 * X0 ** 7 * X1) + (1 - g) * (8 * N0 ** 7 * N1),
         g * (8 * X0 ** 7 * X2 + 28 * X0 ** 6 * X1 ** 2) + (1 - g) * (8 * N0 ** 7 * N2 + 28 * N0 ** 6 * N1 ** 2),
         g * (56 * X0 ** 6 * X1 * X2 + 56 * X0 ** 5 * X1 ** 3) + (1 - g) * (
                 56 * N0 ** 6 * N1 * N2 + 56 * N0 ** 5 * N1 ** 3),
         g * (28 * X0 ** 6 * X2 ** 2 + 168 * X0 ** 5 * X1 ** 2 * X2 + 70 * X0 ** 4 * X1 ** 4) + (1 - g) * (
                 28 * N0 ** 6 * N2 ** 2 + 168 * N0 ** 5 * N1 ** 2 * N2 + 70 * N0 ** 4 * N1 ** 4),
         g * (168 * X0 ** 5 * X1 * X2 ** 2 + 280 * X0 ** 4 * X1 ** 3 * X2 + 56 * X0 ** 3 * X1 ** 5) + (1 - g) * (
                 168 * N0 ** 5 * N1 * N2 ** 2 + 280 * N0 ** 4 * N1 ** 3 * N2 + 56 * N0 ** 3 * N1 ** 5),
         g * (
                 56 * X0 ** 5 * X2 ** 3 + 420 * X0 ** 4 * X1 ** 2 * X2 ** 2 + 280 * X0 ** 3 * X1 ** 4 * X2 + 28 * X0 ** 2 * X1 ** 6) + (
                 1 - g) * (
                 56 * N0 ** 5 * N2 ** 3 + 420 * N0 ** 4 * N1 ** 2 * N2 ** 2 + 280 * N0 ** 3 * N1 ** 4 * N2 + 28 * N0 ** 2 * N1 ** 6),
         g * (
                 280 * X0 ** 4 * X1 * X2 ** 3 + 560 * X0 ** 3 * X1 ** 3 * X2 ** 2 + 168 * X0 ** 2 * X1 ** 5 * X2 + 8 * X0 * X1 ** 7) + (
                 1 - g) * (
                 280 * N0 ** 4 * N1 * N2 ** 3 + 560 * N0 ** 3 * N1 ** 3 * N2 ** 2 + 168 * N0 ** 2 * N1 ** 5 * N2 + 8 * N0 * N1 ** 7),
         g * (
                 70 * X0 ** 4 * X2 ** 4 + 560 * X0 ** 3 * X1 ** 2 * X2 ** 3 + 420 * X0 ** 2 * X1 ** 4 * X2 ** 2 + 56 * X0 * X1 ** 6 * X2 + X1 ** 8) + (
                 1 - g) * (
                 70 * N0 ** 4 * N2 ** 4 + 560 * N0 ** 3 * N1 ** 2 * N2 ** 3 + 420 * N0 ** 2 * N1 ** 4 * N2 ** 2 + 56 * N0 * N1 ** 6 * N2 + N1 ** 8),
         g * (
                 280 * X0 ** 3 * X1 * X2 ** 4 + 560 * X0 ** 2 * X1 ** 3 * X2 ** 3 + 168 * X0 * X1 ** 5 * X2 ** 2 + 8 * X1 ** 7 * X2) + (
                 1 - g) * (
                 280 * N0 ** 3 * N1 * N2 ** 4 + 560 * N0 ** 2 * N1 ** 3 * N2 ** 3 + 168 * N0 * N1 ** 5 * N2 ** 2 + 8 * N1 ** 7 * N2),
         g * (
                 56 * X0 ** 3 * X2 ** 5 + 420 * X0 ** 2 * X1 ** 2 * X2 ** 4 + 280 * X0 * X1 ** 4 * X2 ** 3 + 28 * X1 ** 6 * X2 ** 2) + (
                 1 - g) * (
                 56 * N0 ** 3 * N2 ** 5 + 420 * N0 ** 2 * N1 ** 2 * N2 ** 4 + 280 * N0 * N1 ** 4 * N2 ** 3 + 28 * N1 ** 6 * N2 ** 2),
         g * (168 * X0 ** 2 * X1 * X2 ** 5 + 280 * X0 * X1 ** 3 * X2 ** 4 + 56 * X1 ** 5 * X2 ** 3) + (1 - g) * (
                 168 * N0 ** 2 * N1 * N2 ** 5 + 280 * N0 * N1 ** 3 * N2 ** 4 + 56 * N1 ** 5 * N2 ** 3),
         g * (28 * X0 ** 2 * X2 ** 6 + 168 * X0 * X1 ** 2 * X2 ** 5 + 70 * X1 ** 4 * X2 ** 4) + (1 - g) * (
                 28 * N0 ** 2 * N2 ** 6 + 168 * N0 * N1 ** 2 * N2 ** 5 + 70 * N1 ** 4 * N2 ** 4),
         g * (56 * X0 * X1 * X2 ** 6 + 56 * X1 ** 3 * X2 ** 5) + (1 - g) * (
                 56 * N0 * N1 * N2 ** 6 + 56 * N1 ** 3 * N2 ** 5),
         g * (8 * X0 * X2 ** 7 + 28 * X1 ** 2 * X2 ** 6) + (1 - g) * (8 * N0 * N2 ** 7 + 28 * N1 ** 2 * N2 ** 6),
         g * (8 * X1 * X2 ** 7) + (1 - g) * (8 * N1 * N2 ** 7),
         g * (X2 ** 8) + (1 - g) * (N2 ** 8)]

    return P


def arachidonicISA(e, D, T, N, P):
    # define tracer and naturual abundance isotopomers
    N0 = N[0]
    N1 = N[1]
    N2 = N[2]
    T0 = T[0]
    T1 = T[1]
    T2 = T[2]

    # compute X values
    X0 = D * T0 + (1 - D) * N0
    X1 = D * T1 + (1 - D) * N1
    X2 = D * T2 + (1 - D) * N2

    # compute product isotopmer abundances

    P = [e * (X0 * N0 ** 9) + (1 - e) * (N0 * N0 ** 9),
         e * (X1 * N0 ** 9 + 9 * X0 * N1 * N0 ** 8) + (1 - e) * (N1 * N0 ** 9 + 9 * N0 * N1 * N0 ** 8),
         e * (X2 * N0 ** 9 + 9 * N1 * X1 * N0 ** 8 + 9 * X0 * N2 * N0 ** 8 + 36 * X0 * N1 ** 2 * N0 ** 7) + (1 - e) * (
                 N2 * N0 ** 9 + 9 * N1 * N1 * N0 ** 8 + 9 * N0 * N2 * N0 ** 8 + 36 * N0 * N1 ** 2 * N0 ** 7),
         e * (
                 9 * X1 * N2 * N0 ** 8 + 9 * N1 * X2 * N0 ** 8 + 36 * N1 ** 2 * X1 * N0 ** 7 + 72 * X0 * N1 * N2 * N0 ** 7 + 84 * X0 * N1 ** 3 * N0 ** 6) + (
                 1 - e) * (
                 9 * N1 * N2 * N0 ** 8 + 9 * N1 * N2 * N0 ** 8 + 36 * N1 ** 2 * N1 * N0 ** 7 + 72 * N0 * N1 * N2 * N0 ** 7 + 84 * N0 * N1 ** 3 * N0 ** 6),
         e * (
                 9 * N2 * X2 * N0 ** 8 + 36 * X0 * N2 ** 2 * N0 ** 7 + 72 * N1 * X1 * N2 * N0 ** 7 + 36 * N1 ** 2 * X2 * N0 ** 7 + 84 * N1 ** 3 * X1 * N0 ** 6 + 252 * X0 * N1 ** 2 * N2 * N0 ** 6 + 126 * X0 * N1 ** 4 * N0 ** 5) + (
                 1 - e) * (
                 9 * N2 * N2 * N0 ** 8 + 36 * N0 * N2 ** 2 * N0 ** 7 + 72 * N1 * N1 * N2 * N0 ** 7 + 36 * N1 ** 2 * N2 * N0 ** 7 + 84 * N1 ** 3 * N1 * N0 ** 6 + 252 * N0 * N1 ** 2 * N2 * N0 ** 6 + 126 * N0 * N1 ** 4 * N0 ** 5),
         e * (
                 36 * X1 * N2 ** 2 * N0 ** 7 + 72 * N1 * N2 * X2 * N0 ** 7 + 252 * X0 * N1 * N2 ** 2 * N0 ** 6 + 252 * N1 ** 2 * X1 * N2 * N0 ** 6 + 84 * N1 ** 3 * X2 * N0 ** 6 + 126 * N1 ** 4 * X1 * N0 ** 5 + 504 * X0 * N1 ** 3 * N2 * N0 ** 5 + 126 * X0 * N1 ** 5 * N0 ** 4) + (
                 1 - e) * (
                 36 * N1 * N2 ** 2 * N0 ** 7 + 72 * N1 * N2 * N2 * N0 ** 7 + 252 * N0 * N1 * N2 ** 2 * N0 ** 6 + 252 * N1 ** 2 * N1 * N2 * N0 ** 6 + 84 * N1 ** 3 * N2 * N0 ** 6 + 126 * N1 ** 4 * N1 * N0 ** 5 + 504 * N0 * N1 ** 3 * N2 * N0 ** 5 + 126 * N0 * N1 ** 5 * N0 ** 4),
         e * (
                 36 * N2 ** 2 * X2 * N0 ** 7 + 84 * X0 * N2 ** 3 * N0 ** 6 + 126 * N1 ** 5 * X1 * N0 ** 4 + 630 * X0 * N1 ** 4 * N2 * N0 ** 4 + 252 * N1 * X1 * N2 ** 2 * N0 ** 6 + 252 * N1 ** 2 * N2 * X2 * N0 ** 6 + 84 * X0 * N1 ** 6 * N0 ** 3 + 756 * X0 * N1 ** 2 * N2 ** 2 * N0 ** 5 + 504 * N1 ** 3 * X1 * N2 * N0 ** 5 + 126 * N1 ** 4 * X2 * N0 ** 5) + (
                 1 - e) * (
                 36 * N2 ** 2 * N2 * N0 ** 7 + 84 * N0 * N2 ** 3 * N0 ** 6 + 126 * N1 ** 5 * N1 * N0 ** 4 + 630 * N0 * N1 ** 4 * N2 * N0 ** 4 + 252 * N1 * N1 * N2 ** 2 * N0 ** 6 + 252 * N1 ** 2 * N2 * N2 * N0 ** 6 + 84 * N0 * N1 ** 6 * N0 ** 3 + 756 * N0 * N1 ** 2 * N2 ** 2 * N0 ** 5 + 504 * N1 ** 3 * N1 * N2 * N0 ** 5 + 126 * N1 ** 4 * N2 * N0 ** 5),
         e * (
                 36 * X0 * N1 ** 7 * N0 ** 2 + 1260 * X0 * N1 ** 3 * N2 ** 2 * N0 ** 4 + 84 * X1 * N2 ** 3 * N0 ** 6 + 630 * N1 ** 4 * X1 * N2 * N0 ** 4 + 126 * N1 ** 5 * X2 * N0 ** 4 + 252 * N1 * N2 ** 2 * X2 * N0 ** 6 + 504 * X0 * N1 * N2 ** 3 * N0 ** 5 + 756 * N1 ** 2 * X1 * N2 ** 2 * N0 ** 5 + 84 * N1 ** 6 * X1 * N0 ** 3 + 504 * X0 * N1 ** 5 * N2 * N0 ** 3 + 504 * N1 ** 3 * N2 * X2 * N0 ** 5) + (
                 1 - e) * (
                 36 * N0 * N1 ** 7 * N0 ** 2 + 1260 * N0 * N1 ** 3 * N2 ** 2 * N0 ** 4 + 84 * N1 * N2 ** 3 * N0 ** 6 + 630 * N1 ** 4 * N1 * N2 * N0 ** 4 + 126 * N1 ** 5 * N2 * N0 ** 4 + 252 * N1 * N2 ** 2 * N2 * N0 ** 6 + 504 * N0 * N1 * N2 ** 3 * N0 ** 5 + 756 * N1 ** 2 * N1 * N2 ** 2 * N0 ** 5 + 84 * N1 ** 6 * N1 * N0 ** 3 + 504 * N0 * N1 ** 5 * N2 * N0 ** 3 + 504 * N1 ** 3 * N2 * N2 * N0 ** 5),
         e * (
                 1260 * X0 * N1 ** 2 * N2 ** 3 * N0 ** 4 + 1260 * N1 ** 3 * X1 * N2 ** 2 * N0 ** 4 + 84 * N2 ** 3 * X2 * N0 ** 6 + 630 * N1 ** 4 * N2 * X2 * N0 ** 4 + 36 * N1 ** 7 * X1 * N0 ** 2 + 252 * X0 * N1 ** 6 * N2 * N0 ** 2 + 126 * X0 * N2 ** 4 * N0 ** 5 + 504 * N1 * X1 * N2 ** 3 * N0 ** 5 + 1260 * X0 * N1 ** 4 * N2 ** 2 * N0 ** 3 + 9 * X0 * N1 ** 8 * N0 + 504 * N1 ** 5 * X1 * N2 * N0 ** 3 + 756 * N1 ** 2 * N2 ** 2 * X2 * N0 ** 5 + 84 * N1 ** 6 * X2 * N0 ** 3) + (
                 1 - e) * (
                 1260 * N0 * N1 ** 2 * N2 ** 3 * N0 ** 4 + 1260 * N1 ** 3 * N1 * N2 ** 2 * N0 ** 4 + 84 * N2 ** 3 * N2 * N0 ** 6 + 630 * N1 ** 4 * N2 * N2 * N0 ** 4 + 36 * N1 ** 7 * N1 * N0 ** 2 + 252 * N0 * N1 ** 6 * N2 * N0 ** 2 + 126 * N0 * N2 ** 4 * N0 ** 5 + 504 * N1 * N1 * N2 ** 3 * N0 ** 5 + 1260 * N0 * N1 ** 4 * N2 ** 2 * N0 ** 3 + 9 * N0 * N1 ** 8 * N0 + 504 * N1 ** 5 * N1 * N2 * N0 ** 3 + 756 * N1 ** 2 * N2 ** 2 * N2 * N0 ** 5 + 84 * N1 ** 6 * N2 * N0 ** 3),
         e * (
                 630 * X0 * N1 * N2 ** 4 * N0 ** 4 + 504 * N1 ** 5 * N2 * X2 * N0 ** 3 + 1260 * N1 ** 2 * X1 * N2 ** 3 * N0 ** 4 + 9 * N1 ** 8 * X1 * N0 + 72 * X0 * N1 ** 7 * N2 * N0 + 1260 * N1 ** 3 * N2 ** 2 * X2 * N0 ** 4 + 756 * X0 * N1 ** 5 * N2 ** 2 * N0 ** 2 + 252 * N1 ** 6 * X1 * N2 * N0 ** 2 + 126 * X1 * N2 ** 4 * N0 ** 5 + 36 * N1 ** 7 * X2 * N0 ** 2 + X0 * N1 ** 9 + 252 * N1 * N2 ** 6 * X2 * N0 ** 2 + 1680 * X0 * N1 ** 3 * N2 ** 3 * N0 ** 3 + 1260 * N1 ** 4 * X1 * N2 ** 2 * N0 ** 3 + 504 * N1 * N2 ** 3 * X2 * N0 ** 5) + (
                 1 - e) * (
                 630 * N0 * N1 * N2 ** 4 * N0 ** 4 + 504 * N1 ** 5 * N2 * N2 * N0 ** 3 + 1260 * N1 ** 2 * N1 * N2 ** 3 * N0 ** 4 + 9 * N1 ** 8 * N1 * N0 + 72 * N0 * N1 ** 7 * N2 * N0 + 1260 * N1 ** 3 * N2 ** 2 * N2 * N0 ** 4 + 756 * N0 * N1 ** 5 * N2 ** 2 * N0 ** 2 + 252 * N1 ** 6 * N1 * N2 * N0 ** 2 + 126 * N1 * N2 ** 4 * N0 ** 5 + 36 * N1 ** 7 * N2 * N0 ** 2 + N0 * N1 ** 9 + 252 * N1 * N2 ** 6 * N2 * N0 ** 2 + 1680 * N0 * N1 ** 3 * N2 ** 3 * N0 ** 3 + 1260 * N1 ** 4 * N1 * N2 ** 2 * N0 ** 3 + 504 * N1 * N2 ** 3 * N2 * N0 ** 5),
         e * (
                 126 * X0 * N2 ** 5 * N0 ** 4 + 1260 * N1 ** 4 * N2 ** 2 * X2 * N0 ** 3 + 630 * N1 * X1 * N2 ** 4 * N0 ** 4 + N1 ** 9 * X1 + 9 * X0 * N1 ** 8 * N2 + 252 * X0 * N1 ** 6 * N2 ** 2 * N0 + 72 * N1 ** 7 * X1 * N2 * N0 + 9 * N1 ** 8 * X2 * N0 + 1260 * X0 * N1 ** 4 * N2 ** 3 * N0 ** 2 + 1260 * N1 ** 2 * N2 ** 3 * X2 * N0 ** 4 + 756 * N1 ** 5 * X1 * N2 ** 2 * N0 ** 2 + 1260 * X0 * N1 ** 2 * N2 ** 4 * N0 ** 3 + 1680 * N1 ** 3 * X1 * N2 ** 3 * N0 ** 3 + 252 * N1 ** 6 * N2 * X2 * N0 ** 2 + 126 * N2 ** 4 * X2 * N0 ** 5) + (
                 1 - e) * (
                 126 * N0 * N2 ** 5 * N0 ** 4 + 1260 * N1 ** 4 * N2 ** 2 * N2 * N0 ** 3 + 630 * N1 * N1 * N2 ** 4 * N0 ** 4 + N1 ** 9 * N1 + 9 * N0 * N1 ** 8 * N2 + 252 * N0 * N1 ** 6 * N2 ** 2 * N0 + 72 * N1 ** 7 * N1 * N2 * N0 + 9 * N1 ** 8 * N2 * N0 + 1260 * N0 * N1 ** 4 * N2 ** 3 * N0 ** 2 + 1260 * N1 ** 2 * N2 ** 3 * N2 * N0 ** 4 + 756 * N1 ** 5 * N1 * N2 ** 2 * N0 ** 2 + 1260 * N0 * N1 ** 2 * N2 ** 4 * N0 ** 3 + 1680 * N1 ** 3 * N1 * N2 ** 3 * N0 ** 3 + 252 * N1 ** 6 * N2 * N2 * N0 ** 2 + 126 * N2 ** 4 * N2 * N0 ** 5),
         e * (
                 126 * X1 * N2 ** 5 * N0 ** 4 + 1680 * N1 ** 3 * N2 ** 3 * X2 * N0 ** 3 + 36 * X0 * N1 ** 7 * N2 ** 2 + 504 * X0 * N1 ** 5 * N2 ** 3 * N0 + 9 * N1 ** 8 * X1 * N2 + N1 ** 9 * X2 + 252 * N1 ** 6 * X1 * N2 ** 2 * N0 + 1260 * X0 * N1 ** 3 * N2 ** 4 * N0 ** 2 + 630 * N1 * N2 ** 4 * X2 * N0 ** 4 + 1260 * N1 ** 4 * X1 * N2 ** 3 * N0 ** 2 + 504 * X0 * N1 * N2 ** 5 * N0 ** 3 + 72 * N1 ** 7 * N2 * X2 * N0 + 1260 * N1 ** 2 * X1 * N2 ** 4 * N0 ** 3 + 756 * N1 ** 5 * N2 ** 2 * X2 * N0 ** 2) + (
                 1 - e) * (
                 126 * N1 * N2 ** 5 * N0 ** 4 + 1680 * N1 ** 3 * N2 ** 3 * N2 * N0 ** 3 + 36 * N0 * N1 ** 7 * N2 ** 2 + 504 * N0 * N1 ** 5 * N2 ** 3 * N0 + 9 * N1 ** 8 * N1 * N2 + N1 ** 9 * N2 + 252 * N1 ** 6 * N1 * N2 ** 2 * N0 + 1260 * N0 * N1 ** 3 * N2 ** 4 * N0 ** 2 + 630 * N1 * N2 ** 4 * N2 * N0 ** 4 + 1260 * N1 ** 4 * N1 * N2 ** 3 * N0 ** 2 + 504 * N0 * N1 * N2 ** 5 * N0 ** 3 + 72 * N1 ** 7 * N2 * N2 * N0 + 1260 * N1 ** 2 * N1 * N2 ** 4 * N0 ** 3 + 756 * N1 ** 5 * N2 ** 2 * N2 * N0 ** 2),
         e * (
                 36 * N1 ** 7 * X1 * N2 ** 2 + 630 * X0 * N1 ** 4 * N2 ** 4 * N0 + 504 * N1 ** 5 * X1 * N2 ** 3 * N0 + 756 * X0 * N1 ** 2 * N2 ** 5 * N0 ** 2 + 126 * N2 ** 5 * X2 * N0 ** 4 + 1260 * N1 ** 3 * X1 * N2 ** 4 * N0 ** 2 + 9 * N1 ** 8 * N2 * X2 + 84 * X0 * N2 ** 6 * N0 ** 3 + 252 * N1 ** 6 * N2 ** 2 * X2 * N0 + 504 * N1 * X1 * N2 ** 5 * N0 ** 3 + 1260 * N1 ** 4 * N2 ** 3 * X2 * N0 ** 2 + 84 * X0 * N1 ** 6 * N2 ** 3 + 1260 * N1 ** 2 * N2 ** 4 * X2 * N0 ** 3) + (
                 1 - e) * (
                 36 * N1 ** 7 * N1 * N2 ** 2 + 630 * N0 * N1 ** 4 * N2 ** 4 * N0 + 504 * N1 ** 5 * N1 * N2 ** 3 * N0 + 756 * N0 * N1 ** 2 * N2 ** 5 * N0 ** 2 + 126 * N2 ** 5 * N2 * N0 ** 4 + 1260 * N1 ** 3 * N1 * N2 ** 4 * N0 ** 2 + 9 * N1 ** 8 * N2 * N2 + 84 * N0 * N2 ** 6 * N0 ** 3 + 252 * N1 ** 6 * N2 ** 2 * N2 * N0 + 504 * N1 * N1 * N2 ** 5 * N0 ** 3 + 1260 * N1 ** 4 * N2 ** 3 * N2 * N0 ** 2 + 84 * N0 * N1 ** 6 * N2 ** 3 + 1260 * N1 ** 2 * N2 ** 4 * N2 * N0 ** 3),
         e * (
                 84 * N1 ** 6 * X1 * N2 ** 3 + 504 * X0 * N1 ** 3 * N2 ** 5 * N0 + 630 * N1 ** 4 * X1 * N2 ** 4 * N0 + 252 * X0 * N1 * N2 ** 6 * N0 ** 2 + 756 * N1 ** 2 * X1 * N2 ** 5 * N0 ** 2 + 36 * N1 ** 7 * N2 ** 2 * X2 + 504 * N1 ** 5 * N2 ** 3 * X2 * N0 + 1260 * N1 ** 3 * N2 ** 4 * X2 * N0 ** 2 + 126 * X0 * N1 ** 5 * N2 ** 4 + 504 * N1 * N2 ** 5 * X2 * N0 ** 3 + 84 * X1 * N2 ** 6 * N0 ** 3) + (
                 1 - e) * (
                 84 * N1 ** 6 * N1 * N2 ** 3 + 504 * N0 * N1 ** 3 * N2 ** 5 * N0 + 630 * N1 ** 4 * N1 * N2 ** 4 * N0 + 252 * N0 * N1 * N2 ** 6 * N0 ** 2 + 756 * N1 ** 2 * N1 * N2 ** 5 * N0 ** 2 + 36 * N1 ** 7 * N2 ** 2 * N2 + 504 * N1 ** 5 * N2 ** 3 * N2 * N0 + 1260 * N1 ** 3 * N2 ** 4 * N2 * N0 ** 2 + 126 * N0 * N1 ** 5 * N2 ** 4 + 504 * N1 * N2 ** 5 * N2 * N0 ** 3 + 84 * N1 * N2 ** 6 * N0 ** 3),
         e * (
                 504 * N1 ** 3 * X1 * N2 ** 5 * N0 + 36 * X0 * N2 ** 7 * N0 ** 2 + 252 * N1 * X1 * N2 ** 6 * N0 ** 2 + 84 * N1 ** 6 * N2 ** 3 * X2 + 630 * N1 ** 4 * N2 ** 4 * X2 * N0 + 756 * N1 ** 2 * N2 ** 5 * X2 * N0 ** 2 + 126 * X0 * N1 ** 4 * N2 ** 5 + 84 * N2 ** 6 * X2 * N0 ** 3 + 252 * X0 * N1 ** 2 * N2 ** 6 * N0 + 126 * N1 ** 5 * X1 * N2 ** 4) + (
                 1 - e) * (
                 504 * N1 ** 3 * N1 * N2 ** 5 * N0 + 36 * N0 * N2 ** 7 * N0 ** 2 + 252 * N1 * N1 * N2 ** 6 * N0 ** 2 + 84 * N1 ** 6 * N2 ** 3 * N2 + 630 * N1 ** 4 * N2 ** 4 * N2 * N0 + 756 * N1 ** 2 * N2 ** 5 * N2 * N0 ** 2 + 126 * N0 * N1 ** 4 * N2 ** 5 + 84 * N2 ** 6 * N2 * N0 ** 3 + 252 * N0 * N1 ** 2 * N2 ** 6 * N0 + 126 * N1 ** 5 * N1 * N2 ** 4),
         e * (
                 252 * N1 ** 2 * X1 * N2 ** 6 * N0 + 36 * X1 * N2 ** 7 * N0 ** 2 + 504 * N1 ** 3 * N2 ** 5 * X2 * N0 + 84 * X0 * N1 ** 3 * N2 ** 6 + 72 * X0 * N1 * N2 ** 7 * N0 + 126 * N1 ** 4 * X1 * N2 ** 5 + 126 * N1 ** 5 * N2 ** 4 * X2) + (
                 1 - e) * (
                 252 * N1 ** 2 * N1 * N2 ** 6 * N0 + 36 * N1 * N2 ** 7 * N0 ** 2 + 504 * N1 ** 3 * N2 ** 5 * N2 * N0 + 84 * N0 * N1 ** 3 * N2 ** 6 + 72 * N0 * N1 * N2 ** 7 * N0 + 126 * N1 ** 4 * N1 * N2 ** 5 + 126 * N1 ** 5 * N2 ** 4 * N2),
         e * (
                 126 * N1 ** 4 * N2 ** 5 * X2 + 252 * N1 ** 2 * N2 ** 6 * X2 * N0 + 36 * N2 ** 7 * X2 * N0 ** 2 + 36 * X0 * N1 ** 2 * N2 ** 7 + 9 * X0 * N2 ** 8 * N0 + 84 * N1 ** 3 * X1 * N2 ** 6 + 72 * N1 * X1 * N2 ** 7 * N0) + (
                 1 - e) * (
                 126 * N1 ** 4 * N2 ** 5 * N2 + 252 * N1 ** 2 * N2 ** 6 * N2 * N0 + 36 * N2 ** 7 * N2 * N0 ** 2 + 36 * N0 * N1 ** 2 * N2 ** 7 + 9 * N0 * N2 ** 8 * N0 + 84 * N1 ** 3 * N1 * N2 ** 6 + 72 * N1 * N1 * N2 ** 7 * N0),
         e * (
                 84 * N1 ** 3 * N2 ** 6 * X2 + 72 * N1 * N2 ** 7 * X2 * N0 + 9 * X0 * N1 * N2 ** 8 + 36 * N1 ** 2 * X1 * N2 ** 7 + 9 * X1 * N2 ** 8 * N0) + (
                 1 - e) * (
                 84 * N1 ** 3 * N2 ** 6 * N2 + 72 * N1 * N2 ** 7 * N2 * N0 + 9 * N0 * N1 * N2 ** 8 + 36 * N1 ** 2 * N1 * N2 ** 7 + 9 * N1 * N2 ** 8 * N0),
         e * (36 * N1 ** 2 * N2 ** 7 * X2 + 9 * N2 ** 8 * X2 * N0 + X0 * N2 ** 9 + 9 * N1 * X1 * N2 ** 8) + (1 - e) * (
                 36 * N1 ** 2 * N2 ** 7 * N2 + 9 * N2 ** 8 * N2 * N0 + N0 * N2 ** 9 + 9 * N1 * N1 * N2 ** 8),
         e * (9 * N1 * N2 ** 8 * X2 + X1 * N2 ** 9) + (1 - e) * (9 * N1 * N2 ** 8 * N2 + N1 * N2 ** 9),
         e * (N2 ** 9 * X2) + (1 - e) * (N2 ** 9 * N2)]

    return P


def dhaISA(e, D, T, N, P):
    # define tracer and naturual abundance isotopomers
    N0 = N[0]
    N1 = N[1]
    N2 = N[2]
    T0 = T[0]
    T1 = T[1]
    T2 = T[2]

    # compute X values
    X0 = D * T0 + (1 - D) * N0
    X1 = D * T1 + (1 - D) * N1
    X2 = D * T2 + (1 - D) * N2

    # compute product isotopmer abundances
    P = [e * (N0 ** 9 * X0 ** 2) + (1 - e) * (N0 ** 9 * N0 ** 2),
         e * (9 * N0 ** 8 * X0 ** 2 * N1 + 2 * N0 ** 9 * X0 * X1) + (1 - e) * (
                 9 * N0 ** 8 * N0 ** 2 * N1 + 2 * N0 ** 9 * N0 * N1),
         e * (
                 36 * N0 ** 7 * X0 ** 2 * N1 ** 2 + 18 * N0 ** 8 * X0 * N1 * X1 + N0 ** 9 * X1 ** 2 + 9 * N0 ** 8 * X0 ** 2 * N2 + 2 * N0 ** 9 * X0 * X2) + (
                 1 - e) * (
                 36 * N0 ** 7 * N0 ** 2 * N1 ** 2 + 18 * N0 ** 8 * N0 * N1 * N1 + N0 ** 9 * N1 ** 2 + 9 * N0 ** 8 * N0 ** 2 * N2 + 2 * N0 ** 9 * N0 * N2),
         e * (
                 84 * N0 ** 6 * X0 ** 2 * N1 ** 3 + 72 * N0 ** 7 * X0 * N1 ** 2 * X1 + 9 * N0 ** 8 * N1 * X1 ** 2 + 72 * N0 ** 7 * X0 ** 2 * N1 * N2 + 18 * N0 ** 8 * X0 * X1 * N2 + 18 * N0 ** 8 * X0 * N1 * X2 + 2 * N0 ** 9 * X1 * X2) + (
                 1 - e) * (
                 84 * N0 ** 6 * N0 ** 2 * N1 ** 3 + 72 * N0 ** 7 * N0 * N1 ** 2 * N1 + 9 * N0 ** 8 * N1 * N1 ** 2 + 72 * N0 ** 7 * N0 ** 2 * N1 * N2 + 18 * N0 ** 8 * N0 * N1 * N2 + 18 * N0 ** 8 * N0 * N1 * N2 + 2 * N0 ** 9 * N1 * N2),
         e * (
                 126 * N0 ** 5 * X0 ** 2 * N1 ** 4 + 168 * N0 ** 6 * X0 * N1 ** 3 * X1 + 36 * N0 ** 7 * N1 ** 2 * X1 ** 2 + 252 * N0 ** 6 * X0 ** 2 * N1 ** 2 * N2 + 144 * N0 ** 7 * X0 * N1 * X1 * N2 + 9 * N0 ** 8 * X1 ** 2 * N2 + 36 * N0 ** 7 * X0 ** 2 * N2 ** 2 + 72 * N0 ** 7 * X0 * N1 ** 2 * X2 + 18 * N0 ** 8 * N1 * X1 * X2 + 18 * N0 ** 8 * X0 * N2 * X2 + N0 ** 9 * X2 ** 2) + (
                 1 - e) * (
                 126 * N0 ** 5 * N0 ** 2 * N1 ** 4 + 168 * N0 ** 6 * N0 * N1 ** 3 * N1 + 36 * N0 ** 7 * N1 ** 2 * N1 ** 2 + 252 * N0 ** 6 * N0 ** 2 * N1 ** 2 * N2 + 144 * N0 ** 7 * N0 * N1 * N1 * N2 + 9 * N0 ** 8 * N1 ** 2 * N2 + 36 * N0 ** 7 * N0 ** 2 * N2 ** 2 + 72 * N0 ** 7 * N0 * N1 ** 2 * N2 + 18 * N0 ** 8 * N1 * N1 * N2 + 18 * N0 ** 8 * N0 * N2 * N2 + N0 ** 9 * N2 ** 2),
         e * (
                 126 * N0 ** 4 * X0 ** 2 * N1 ** 5 + 252 * N0 ** 5 * X0 * N1 ** 4 * X1 + 84 * N0 ** 6 * N1 ** 3 * X1 ** 2 + 504 * N0 ** 5 * X0 ** 2 * N1 ** 3 * N2 + 504 * N0 ** 6 * X0 * N1 ** 2 * X1 * N2 + 72 * N0 ** 7 * N1 * X1 ** 2 * N2 + 252 * N0 ** 6 * X0 ** 2 * N1 * N2 ** 2 + 72 * N0 ** 7 * X0 * X1 * N2 ** 2 + 168 * N0 ** 6 * X0 * N1 ** 3 * X2 + 72 * N0 ** 7 * N1 ** 2 * X1 * X2 + 144 * N0 ** 7 * X0 * N1 * N2 * X2 + 18 * N0 ** 8 * X1 * N2 * X2 + 9 * N0 ** 8 * N1 * X2 ** 2) + (
                 1 - e) * (
                 126 * N0 ** 4 * N0 ** 2 * N1 ** 5 + 252 * N0 ** 5 * N0 * N1 ** 4 * N1 + 84 * N0 ** 6 * N1 ** 3 * N1 ** 2 + 504 * N0 ** 5 * N0 ** 2 * N1 ** 3 * N2 + 504 * N0 ** 6 * N0 * N1 ** 2 * N1 * N2 + 72 * N0 ** 7 * N1 * N1 ** 2 * N2 + 252 * N0 ** 6 * N0 ** 2 * N1 * N2 ** 2 + 72 * N0 ** 7 * N0 * N1 * N2 ** 2 + 168 * N0 ** 6 * N0 * N1 ** 3 * N2 + 72 * N0 ** 7 * N1 ** 2 * N1 * N2 + 144 * N0 ** 7 * N0 * N1 * N2 * N2 + 18 * N0 ** 8 * N1 * N2 * N2 + 9 * N0 ** 8 * N1 * N2 ** 2),
         e * (
                 84 * N0 ** 3 * X0 ** 2 * N1 ** 6 + 252 * N0 ** 4 * X0 * N1 ** 5 * X1 + 126 * N0 ** 5 * N1 ** 4 * X1 ** 2 + 630 * N0 ** 4 * X0 ** 2 * N1 ** 4 * N2 + 1008 * N0 ** 5 * X0 * N1 ** 3 * X1 * N2 + 252 * N0 ** 6 * N1 ** 2 * X1 ** 2 * N2 + 756 * N0 ** 5 * X0 ** 2 * N1 ** 2 * N2 ** 2 + 504 * N0 ** 6 * X0 * N1 * X1 * N2 ** 2 + 36 * N0 ** 7 * X1 ** 2 * N2 ** 2 + 84 * N0 ** 6 * X0 ** 2 * N2 ** 3 + 252 * N0 ** 5 * X0 * N1 ** 4 * X2 + 168 * N0 ** 6 * N1 ** 3 * X1 * X2 + 504 * N0 ** 6 * X0 * N1 ** 2 * N2 * X2 + 144 * N0 ** 7 * N1 * X1 * N2 * X2 + 72 * N0 ** 7 * X0 * N2 ** 2 * X2 + 36 * N0 ** 7 * N1 ** 2 * X2 ** 2 + 9 * N0 ** 8 * N2 * X2 ** 2) + (
                 1 - e) * (
                 84 * N0 ** 3 * N0 ** 2 * N1 ** 6 + 252 * N0 ** 4 * N0 * N1 ** 5 * N1 + 126 * N0 ** 5 * N1 ** 4 * N1 ** 2 + 630 * N0 ** 4 * N0 ** 2 * N1 ** 4 * N2 + 1008 * N0 ** 5 * N0 * N1 ** 3 * N1 * N2 + 252 * N0 ** 6 * N1 ** 2 * N1 ** 2 * N2 + 756 * N0 ** 5 * N0 ** 2 * N1 ** 2 * N2 ** 2 + 504 * N0 ** 6 * N0 * N1 * N1 * N2 ** 2 + 36 * N0 ** 7 * N1 ** 2 * N2 ** 2 + 84 * N0 ** 6 * N0 ** 2 * N2 ** 3 + 252 * N0 ** 5 * N0 * N1 ** 4 * N2 + 168 * N0 ** 6 * N1 ** 3 * N1 * N2 + 504 * N0 ** 6 * N0 * N1 ** 2 * N2 * N2 + 144 * N0 ** 7 * N1 * N1 * N2 * N2 + 72 * N0 ** 7 * N0 * N2 ** 2 * N2 + 36 * N0 ** 7 * N1 ** 2 * N2 ** 2 + 9 * N0 ** 8 * N2 * N2 ** 2)]
    P += [0 for _ in range(23 - len(P))]

    P = list(np.array(P) / np.sum(P))

    return P

def correctNaturalAbundance(vec,formula,charge = -1,q=None):
    data = pd.DataFrame(data=vec.reshape(1, -1), index=[0],
                        columns=["No label"] + [str(x + 1) + "C13" for x in range(len(vec) - 1)])
    vec_cor = picor.calc_isotopologue_correction(data, molecule_formula=formula, molecule_charge=charge,resolution_correction=False).values[0]#,resolution=resolution,mz_calibration=res_mz).values[0]

    if type(q) != type(None):
        q.put(0)

    return vec_cor

def convertSpectraArraysToDict(mzs,inten,thresh):
    return {mz:i for mz,i in zip(mzs,inten) if i > thresh}

def convertSpectraAndExtractIntensity(mzs,inten,thresh,targets,ppm,dtype,q=None):
    spec = convertSpectraArraysToDict(mzs,inten,thresh)
    intensities = np.array([extractIntensity(mz,spec,ppm,dtype) for mz in targets])

    if type(q) != type(None):
        q.put(0)

    return intensities



class MSIData():
    def __init__(self,targets,ppm,mass_range = [0,1000],numCores = 1,intensityCutoff = 100):
        """
        Class to manage and process MSI data.
        :param targets: list/ndarray, m/zs of interest
        :param ppm: float, mass tolerance in ppm to extract masses
        :param mass_range: iterable, mass range to consider
        :param numCores: int, number of processor cores to use
        :param intensityCutoff: float, minimum intensity to keep signal in processed data (is not used for TIC)
        """
        self.data_tensor = np.array([])
        self.ppm = ppm
        self.polarity = 0
        self.targets = targets
        self.tic_image = -1
        self.mass_range = mass_range
        self.imageBoundary = -1
        self.numCores = numCores
        self.intensityCutoff = intensityCutoff


    def readimzML(self,filename,dims=None):
        """
        Load data from an imzml file with path filename.
        :param filename: str, path to imzml file
        :param dims: iterable, dimensions of image if imzml dimensions are corrupted. Optional
        :return: None
        """
        p = ImzMLParser(filename)  # load data
        self.polarity = p.polarity
        if type(dims) != type(None):
            p.imzmldict["max count of pixels x"] = dims[0]
            p.imzmldict["max count of pixels y"] = dims[1]

        self.tic_image = getionimage(p, np.mean(self.mass_range), self.mass_range[1] - self.mass_range[0])
        self.data_tensor = np.zeros((len(self.targets),self.tic_image.shape[0],self.tic_image.shape[1]))
        inds = []
        args = []

        for idx, (x,y,_) in enumerate(p.coordinates):
            mzs, intensities = p.getspectrum(idx)
            args.append([mzs,intensities,self.intensityCutoff,self.targets,self.ppm,"centroid"])
            inds.append([y-1,x-1])

        result = startConcurrentTask(convertSpectraAndExtractIntensity,args,self.numCores,"extracting intensities",len(args))
        for [x,y],intensities in zip(inds,result):
            self.data_tensor[:,x,y] = intensities

        self.imageBoundary = np.ones(self.tic_image.shape)

    def readHDIOutput(self,filename,polarity):
        """
        Load data from water HDI .txt output
        :param filename: str, path to .txt file
        :param polarity: str, "negative" or "positive" specifying the polarity of the data
        :return: None
        """

        #read data
        data = [r.strip().split() for r in open(filename, "r").readlines()[3:]]
        data = {(x[0], float(x[1]), float(x[2])): {float(mz): float(i) for mz, i in zip(data[0], x[3:])} for x in data[1:] if
                len(x) > 0}

        #construct df
        data = pd.DataFrame.from_dict(data, orient="index")
        mzs = data.columns.values

        # covert to image coordinates
        xcords = [float(y) for y in list(set([x[2] for x in data.index.values]))]
        ycords = [float(y) for y in list(set([x[1] for x in data.index.values]))]

        # make output array and resize if necessary
        self.data_tensor = np.zeros((len(self.targets),len(xcords), len(ycords)))

        xcords.sort()
        ycords.sort()
        xcordMap = {x: i for x, i in zip(xcords, range(len(xcords)))}
        ycordMap = {x: i for x, i in zip(ycords, range(len(ycords)))}


        # gather images for mzs of interest
        i = 0
        for mz in self.targets:
            # iterate through mzs of interest
            width = self.ppm * mz / 1e6
            mz_start = mz - width
            mz_end = mz + width
            matches = [x for x in mzs if x > mz_start and x < mz_end]
            for index,row in data.iterrows():
                self.data_tensor[i,xcordMap[index[2]],ycordMap[index[1]]] += np.sum(row[matches].values)
            i += 1
        #make tic image
        self.tic_image = np.zeros((len(xcords),len(ycords)))

        for index,row in data.iterrows():
            self.tic_image[xcordMap[index[2]], ycordMap[index[1]]] += np.sum(row[mzs].values)

        self.polarity = polarity
        self.imageBoundary = np.ones(self.tic_image.shape)


    def to_pandas(self):
        """
        reformat data as pandas df
        :return: DataFrame, pandas dataframe with MSI data
        """
        #get dimensions of data
        nrows = self.data_tensor.shape[1]
        ncols = self.data_tensor.shape[2]
        ntotal = nrows * ncols
        df = pd.DataFrame(index=range(ntotal))

        #construct df
        for met, i in zip(self.targets, range(len(self.data_tensor))):
            x = []
            y = []
            ints = []
            for r in range(nrows):
                for c in range(ncols):
                    x.append(c)
                    y.append(r)
                    ints.append(self.data_tensor[i][r][c])
            df[met] = ints

        imageBoundary = []
        tic = []
        for r in range(nrows):
            for c in range(ncols):
                imageBoundary.append(self.imageBoundary[r,c])
                tic.append(self.tic_image[r,c])


        #set coordinates
        df["tic"] = tic
        df["boundary"] = imageBoundary
        df["x"] = x
        df["y"] = y

        df = df[["x", "y"] + list(df.columns.values[:-2])]

        return df

    def from_pandas(self,df,polarity):
        self.polarity = polarity
        targetsFound = [x for x in df.columns.values if x not in ["x","y","tic","boundary"]]
        xdim = len(set(df["x"].values))
        ydim = len(set(df["y"].values))
        self.data_tensor = np.zeros((len(self.targets),ydim,xdim))
        self.imageBoundary = np.zeros((ydim,xdim))
        self.tic_image = np.zeros((ydim,xdim))

        mapper = {mz:[col for col in targetsFound if 1e6 * np.abs(float(col)-mz)/mz < self.ppm] for mz in self.targets}

        for index,row in df.iterrows():
            for x in range(len(self.targets)):
                matches = mapper[self.targets[x]]
                i = np.sum(row[matches].values)
                self.data_tensor[x,int(row["y"]),int(row["x"])] = i
                self.tic_image[int(row["y"]),int(row["x"])] = row["tic"]
                self.imageBoundary[int(row["y"]),int(row["x"])] = row["boundary"]


    def to_imzML(self,outfile):
        """
        Write data as imzml file
        :param outfile: str, path to output file
        :return: None
        """

        #open output file
        output = ImzMLWriter(outfile, polarity=self.polarity)

        #format as df
        df = self.to_pandas()

        #write to output file
        for id,row in df.iterrows():
            mzs = self.targets
            sigs = [row[x] for x in self.targets]
            output.addSpectrum(mzs,sigs,(int(row["x"]),int(row["y"])))

        #close output file
        output.close()

    def segmentImage(self,method="TIC_auto", threshold=0, num_latent=2, dm_method="PCA",fill_holes = True):
        """
        Segment image into sample and background
        :param method: str, method for segmentation ("TIC auto" = find optimal separation between background and foreground based on TIC intensity, "K_means"=use K-means clustering, "TIC_manual"= use a user-defined threshold for segment image based on TIC
        :param threshold: float, intensity threshold for TIC_manual method
        :param num_latent: int, number of latent variables to use in dimensionality reduction prior to clustering when using K_means
        :param dm_method: str, dimensionality reduction method to use with K_means ("PCA" or "TSNE")
        :param fill_holes: bool, True or False depending
        :return:
        """

        # go through all features in dataset

        if method == "TIC_auto":
            # show image and pixel histogram
            plt.figure()
            plt.hist(self.tic_image.flatten())

            # get threshold and mask image
            threshold = skimage.filters.threshold_otsu(self.tic_image)

            imageBoundary = self.tic_image > threshold

            plt.plot([threshold, threshold], [0, 1000])

        elif method == "TIC_manual":
            plt.figure()
            plt.hist(self.tic_image.flatten())
            # get threshold and mask image
            imageBoundary = self.tic_image > threshold

            plt.plot([threshold, threshold], [0, 1000])

        elif method == "K_means":
            kmean = KMeans(2)

            format_data = self.to_pandas()
            xs = format_data["y"].values
            ys = format_data["x"].values
            format_data = format_data[self.targets].to_numpy()

            format_data = imputeRowMin(format_data)
            format_data = np.log2(format_data)

            plt.figure()
            if dm_method == "PCA":
                pca = PCA(n_components=num_latent)
                format_data = pca.fit_transform(format_data)
                plt.xlabel("PC1 (" + str(np.round(100 * pca.explained_variance_ratio_[0], 2)) + "%)")
                plt.ylabel("PC2 (" + str(np.round(100 * pca.explained_variance_ratio_[1], 2)) + "%)")

            elif dm_method == "TSNE":
                tsne = TSNE(n_components=2)
                format_data = tsne.fit_transform(format_data)
                plt.xlabel("t-SNE1")
                plt.ylabel("t-SNE2")

            labels = kmean.fit_predict(format_data)
            group0Int = np.mean(
                [self.tic_image[xs[x], ys[x]] for x in range(len(labels)) if labels[x] < .5])
            group1Int = np.mean(
                [self.tic_image[xs[x], ys[x]] for x in range(len(labels)) if labels[x] > .5])

            if group0Int > group1Int:
                labels = labels < .5

            plt.scatter(format_data[:, 0], format_data[:, 1], c=labels)

            imageBoundary = np.zeros(self.tic_image.shape)
            for x in range(len(labels)):
                if labels[x] > .5:
                    imageBoundary[xs[x], ys[x]] = 1

        if fill_holes: imageBoundary = ndimage.binary_fill_holes(imageBoundary)

        self.imageBoundary = imageBoundary

        self.tic_image = np.multiply(self.tic_image,self.imageBoundary)

        for x in range(len(self.targets)):
            self.data_tensor[x] = np.multiply(self.data_tensor[x],self.imageBoundary)

    def smoothData(self,method,kernal_size):
        # apply moving average filter
        offset = int((kernal_size - 1) / 2)
        height,width = self.tic_image.shape

        result = startConcurrentTask(convolveLayer,[[offset, height, width, self.data_tensor[t], self.imageBoundary, method] for t in
                                            range(len(self.targets))],self.numCores,"Smoothing data",len(self.targets))

        tensorFilt = np.array(result)

        tic_smoothed = convolveLayer(offset,height,width,self.tic_image,self.imageBoundary,method)


        self.data_tensor = tensorFilt
        self.tic_image = tic_smoothed

    def runISA(self,isaModel="flexible",T=[0,0,1],X_image = None):
        c13ab = 0.011  # natural abundance
        N = [(1 - c13ab) ** 2, 2 * (1 - c13ab) * c13ab,
             c13ab ** 2]  # get expected labeling of precursor from natural abundance

        # create data structures to store output
        errs = []
        T_founds = []
        fluxImageG = np.zeros(self.tic_image.shape)
        fluxImageD = np.zeros(self.tic_image.shape)
        fluxImageT0 = np.zeros(self.tic_image.shape)
        fluxImageT1 = np.zeros(self.tic_image.shape)
        fluxImageT2 = np.zeros(self.tic_image.shape)
        P_consider = []
        P_trues = []
        P_preds = []

        # do pixel by pixel ISA
        argList = []
        coords = []

        numCarbons = len(self.data_tensor) - 1
        data = normalizeTensor(self.data_tensor)
        func = getISAEq(numCarbons)

        numFounds = []

        for r in range(self.tic_image.shape[0]):
            for c in range(self.tic_image.shape[1]):
                # get product labeling
                P = data[:, r, c]

                goodInd = [x for x in range(len(P)) if P[x] > 0]

                # if not on background pixel
                if self.imageBoundary[r, c] > .5:
                    # fit ISA
                    numFounds.append(len(goodInd))
                    a = [T, N, P, func, goodInd, .5,False]
                    coords.append((r, c))
                    P_consider.append(P)
                    if isaModel == "dual":
                        a[0] = X_image[:,r,c]
                    argList.append(a)  # np.random.random(1)))

        if isaModel == "flexible":
            eq = ISAFit
        elif isaModel == "classical":
            eq = ISAFit_classical
        elif isaModel == "dual":
            eq = ISAFit_knownT
        else:
            print("ISA model not recognized")
            return -1

        results = startConcurrentTask(eq,argList,self.numCores,"Running ISA",len(argList))

        errors = []
        for (g, D, T_found, err, P_pred), (r, c), P_true in zip(results, coords, P_consider):
            # save results in data structures
            if g > -.0001 and g < 1.1 and all(xx > -0.01 and xx < 1.1 for xx in T_found):
                errs.append(err)
                T_founds.append(T_found)
                P_preds.append(P_pred)
                P_trues.append(P_true)

                fluxImageG[r, c] = g
                fluxImageD[r, c] = D
                fluxImageT0[r, c] = T_found[0]
                fluxImageT1[r, c] = T_found[1]
                fluxImageT2[r, c] = T_found[2]
            else:
                errors.append(P_true)

        return fluxImageG,fluxImageD,fluxImageT0,fluxImageT1,fluxImageT2,T_founds,P_trues,P_preds,numFounds,errs,errors

    def correctNaturalAbundance(self,formula):
        if self.polarity == "positive": charge = 1
        else: charge = -1
        args = []
        coords = []
        for r in range(self.tic_image.shape[0]):
            for c in range(self.tic_image.shape[1]):
                if self.imageBoundary[r,c] > 0.5:
                    vec = self.data_tensor[:,r,c]
                    args.append([vec,formula,charge])
                    coords.append([r,c])

        results = startConcurrentTask(correctNaturalAbundance,args,self.numCores,"correcting natural abundance",len(args))

        for corr,(x,y) in zip(results,coords):
            self.data_tensor[:,x,y] = corr

def getMzsOfIsotopologues(formula,elementOfInterest = "C"):
    # calculate relevant m/z's
    m0Mz = f = molmass.Formula(formula)  # create formula object
    m0Mz = f.isotope.mass  # get monoisotopcic mass for product ion
    # get number of carbons
    comp = f.composition()
    for row in comp:
        if row[0] == elementOfInterest:
            numCarbons = int(row[1])

    mzsOI = [m0Mz + 1.00336 * x for x in range(numCarbons + 1)]
    return m0Mz,mzsOI,numCarbons

def extractIntensity(mz, spectrum, ppm, mode="profile",q=None):
    if mode == "centroid":
        width = ppm * mz / 1e6
    else:
        width = mz / ppm / 2
    mz_start = mz - width
    mz_end = mz + width
    intensity = np.sum([i for mz, i in spectrum.items() if mz > mz_start and mz < mz_end])

    if type(q) != type(None):
        q.put(0)

    return intensity

def showImage(arr,cmap):
    plt.imshow(arr,cmap=cmap)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

def saveTIF(arr,filename):
    im = Image.fromarray(arr)
    im.save(filename)


def normalizeTensor(tensorFilt):
    normalizedTensor = np.zeros(tensorFilt.shape)
    for r in range(len(tensorFilt[0])):
        for c in range(len(tensorFilt[0][0])):
            sumInt = np.sum(tensorFilt[:, r, c])
            normalizedTensor[:, r, c] = tensorFilt[:, r, c] / max([1, sumInt])
    return normalizedTensor

def getISAEq(numCarbons):
    d = {16: palmitateISA, 14: myristicISA, 18: stearicISA, 20: arachidonicISA, 22: dhaISA}
    return d[numCarbons]

