
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
from scipy.integrate import quad,quad_vec,fixed_quad


# simple image scaling to (nR x nC) size
def scale(im, nR, nC):
    nR0 = len(im)  # source number of rows
    nC0 = len(im[0])  # source number of columns
    blockSizeR = int(np.ceil(nR0 / nR))
    blockSizeC = int(np.ceil(nC0 / nC))

    if nR0 >= nR and nC0 >= nC:

        outarray = np.zeros((nR, nC))

        for r in range(nR):
            for c in range(nC):
                if blockSizeR * (r + 1) > nR0 - 1:
                    stopR = nR0
                else:
                    stopR = blockSizeR * (r + 1)
                if blockSizeC * (c + 1) > nC0 - 1:
                    stopC = nC0
                else:
                    stopC = blockSizeC * (c + 1)

                outarray[r, c] = np.sum(im[blockSizeR * r:stopR, blockSizeC * c:stopC])
        return outarray
    else:
        print("image can only be downsampled")
        return im


def objectiveFunc(t, p, goodInd,params = [],alpha=0,lam=0):

    trel = np.array([t[x] for x in goodInd])
    prel = np.array([p[x] for x in goodInd])

    trel = trel / np.sum(trel)
    prel = prel / np.sum(prel)

    return np.sum(np.square(np.subtract(trel, prel))) + alpha*lam*np.sum(np.abs(params)) + (1-alpha)/2 * lam * np.sum(np.square(params))


def ISAFit(T, N, P, func, goodInd, x_init=np.random.random((1)), plot=False):

    success = False
    initial_params = np.concatenate((x_init, T), axis=None)
    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, func(x[0], 1.0, [xx / np.sum(x[1:]) for xx in x[1:]], N, P), goodInd),
            x0=initial_params,
            )#bounds=[(0, 1) for _ in range(len(x_init) + len(T))])
        if not sol.success:
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    #g, D = sol.x[:2]
    g = sol.x[0]
    D = 1
    T = sol.x[1:]
    T = T/np.sum(T)
    err = sol.fun
    P_pred = func(g, D, T, N, P)
    P_pred = P_pred/np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0
    x_ind = 0
    x_lab = []
    maxY = np.max(np.concatenate((P, P_pred)))
    i = 0
    if plot:
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
    return g, D, T, err, P_pred

def integratedISAFull(G,k1,k2,k3,k4,T,N,numC):
    func = getISAEq(numC)
    ts = np.linspace(0,1,20)
    vals  = np.array([parameterizedIntegrandFull(t,N,G,k1,k2,k3,k4,T,func) for t in ts])
    output = np.array([np.trapz(vals[:,x],ts) for x in range(len(vals[0]))])
    return output

def parameterizedIntegrandFull(t,N,G,k1,k2,k3,k4,T,func):
    val = [1-generalizedExp(t,1-T[0],k2),generalizedExp(t,T[1],k3),generalizedExp(t,T[2],k4)]
    val = val / np.sum(val)
    return np.array(func(full_dgdt(t,G,k1),1.0,val,N,None))

def integrated_X_full(T,k2,k3,k4):
    #ts = np.linspace(0, 1, 100)
    #vals = []
    #for t in ts:
    #    val = [1 - generalizedExp(t, 1 - T[0], k2), generalizedExp(t, T[1], k3), generalizedExp(t, T[2], k4)]
    #    val = val / np.sum(val)
    #    vals.append(val)
    #vals=np.array(vals)
    #output = np.array([np.trapz(vals[:, x], ts) for x in range(len(vals[0]))])

    val = [1 - generalizedExp(1, 1 - T[0], k2), generalizedExp(1, T[1], k3), generalizedExp(1, T[2], k4)]
    output = val / np.sum(val)

    return output


def integratedISA(G,k1,k2,T,N,numC):
    func = getISAEq(numC)
    ts = np.linspace(0,1,20)
    T = T/np.sum(T)
    vals  = np.array([parameterizedIntegrand(t,N,G,k1,k2,T,func) for t in ts])
    output = np.array([np.trapz(vals[:,x],ts) for x in range(len(vals[0]))])
    return output

    #return
    #return quad_vec(lambda t:parameterizedIntegrand(t,N,G,k1,k2,r,func),0,1)[0]
    #return fixed_quad(parameterizedIntegrand, 0,1,n=5,args=(N,G,k1,k2,r,func))[0]

def generalizedExp(t,c,k):
    return c*(1-np.exp(-1*k*t))

def full_g_t(t,G,k):
    return generalizedExp(t,G,k)

def full_dgdt(t,G,k):
    return k*G*np.exp(-1*k*t)

def d_t(t,D,k):
    return generalizedExp(t,D,k)

def parameterizedIntegrand(t,N,G,k1,k2,T,func):
    return np.array(func(full_dgdt(t,G,k1),d_t(t,1,k2),T,N,None))

def integrated_X(T,N,k2):
    #ts = np.linspace(0,1,100)
    #vals = np.array([d_t(t,1,k2)*np.array(T)+(1-d_t(t,1,k2)) * np.array(N) for t in ts])
    #output = np.array([np.trapz(vals[:,x],ts) for x in range(len(vals[0]))])
    #return output
    t=1
    return d_t(t, 1, k2) * np.array(T) + (1 - d_t(t, 1, k2)) * np.array(N)



def ISAFit_nonSS(T, N, P, numC, goodInd, x_init=np.random.random((3)), plot=False):

    success = False

    initial_params = np.concatenate((x_init, T), axis=None)
    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, integratedISA(x[0],x[1],x[2],[x[3],x[4],x[5]],N,numC), goodInd,[x[0],x[4]/np.sum(x[3:]),x[5]/np.sum(x[3:])],alpha=0,lam=1e-2),
            x0=initial_params,
            bounds=[(0, m) for m in [1,np.inf,np.inf,np.inf,np.inf,np.inf]])
        if not sol.success:
            print("failed")
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    #g, D = sol.x[:2]
    g = full_g_t(1,sol.x[0],sol.x[1])
    D = d_t(1,1,sol.x[2])
    T = sol.x[3:]
    T = T/np.sum(T)


    err = sol.fun
    P_pred = integratedISA(sol.x[0],sol.x[1],sol.x[2],T,N,numC)
    P_pred = P_pred/np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0
    x_ind = 0
    x_lab = []
    maxY = np.max(np.concatenate((P, P_pred)))
    i = 0
    if plot:
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

    T = integrated_X(T,N,sol.x[2])
    return g, D, T, err, P_pred

def ISAFit_nonSS_full(T, N, P, numC, goodInd, x_init=np.random.random((5)), plot=False):

    success = False

    initial_params = np.concatenate((x_init, T), axis=None)
    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, integratedISAFull(x[0],x[1],x[2],x[3],x[4],x[5:],N,numC), goodInd),
            x0=initial_params,
            bounds=[(0, m) for m in [1,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]])
        if not sol.success:
            print("failed")
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    #g, D = sol.x[:2]
    g = full_g_t(1,sol.x[0],sol.x[1])
    D = 1.0
    T = sol.x[5:]
    #T[0] = 1-T[0]
    T = T/np.sum(T)

    err = sol.fun
    P_pred = integratedISAFull(sol.x[0],sol.x[1],sol.x[2],sol.x[3],sol.x[4],sol.x[5:],N,numC)
    P_pred = P_pred/np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0
    x_ind = 0
    x_lab = []
    maxY = np.max(np.concatenate((P, P_pred)))
    i = 0
    if plot:
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

    T = integrated_X_full(T,sol.x[2],sol.x[3],sol.x[4])

    return g, D, T, err, P_pred



def ISAFit_classical(T, N, P, func, goodInd, x_init=np.random.random((2)), plot=False):

    success = False
    initial_params = x_init
    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, func(x[0], x[1], T, N, P), goodInd),
            x0=initial_params,
            )#bounds=[(0, 1) for _ in range(len(x_init) + len(T))])
        if not sol.success:
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    #g, D = sol.x[:2]
    g = sol.x[0]
    D = sol.x[1]

    err = sol.fun
    P_pred = func(g, D, T, N, P)
    P_pred = P_pred/np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0
    x_ind = 0
    x_lab = []
    maxY = np.max(np.concatenate((P, P_pred)))
    i = 0
    if plot:
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
    return g, D, T, err, P_pred


def ISAFit_knownT(T, N, P, func, goodInd, x_init=np.random.random((2, 1)), plot=False):
    sol = opt.minimize(
        lambda x: objectiveFunc(P, func(x[0], 1.0, T, N, P), goodInd),
        x0=x_init)
    g, D = sol.x[:2]
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
    maxY = np.max(np.concatenate((P, P_pred)))

    if plot:
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
    return g, D, T, err, P_pred

def convolveLayer(offset,height,width,layer,imageBoundary,method="MA"):
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
        tensorFilt = ndimage.gaussian_filter(layer,offset)

    return tensorFilt

#imput matrix with half feature minimum
def imputeRowMin(arr,alt_min=2):
    #find the minimum non-zero value of each compound
    numImp = 0
    max_vals = []
    for c in arr.transpose():
        tmp = [x for x in c if x > alt_min]
        if len(tmp) > 0:
            val = np.min(tmp)
        else:
            val = 2*alt_min
        max_vals.append(val)
    #impute values

    data_imp = np.zeros((len(arr),len(arr[0])))

    for c in range(len(arr[0])):
        for r in range(len(arr)):
            if arr[r,c] > alt_min:
                if np.isinf(arr[r,c]) or np.isnan(arr[r,c]):
                    print("bad", arr[r,c])
                data_imp[r,c] = arr[r,c]
            else:
                data_imp[r,c] = max_vals[c]/2
                numImp += 1
            if data_imp[r,c] < 1e-3:
                data_imp[r,c] = alt_min
                numImp += 1
    return data_imp

def segmentImage(data,height,width,mzs,colormap,method="TIC_auto",threshold=0,num_latent=2,dm_method="PCA"):
    # go through all features in dataset
    allFeatTensor = np.array([getImage(data, x, height, width) for x in mzs])

    sumImage = np.sum(allFeatTensor, axis=0)
    plt.imshow(sumImage, cmap=colormap)
    plt.colorbar()
    plt.figure()

    if method=="TIC_auto":
        # show image and pixel histogram
        plt.hist(sumImage.flatten())

        # get threshold and mask image
        threshold = skimage.filters.threshold_otsu(sumImage)

        imageBoundary = sumImage > threshold

        plt.plot([threshold, threshold], [0, 1000])
    elif method == "TIC_manual":
        plt.hist(sumImage.flatten())
        # get threshold and mask image
        imageBoundary = sumImage > threshold

        plt.plot([threshold, threshold], [0, 1000])

    elif method == "K_means":
        kmean = KMeans(2)
        format_data = []
        ind_mapper = {}
        for r in range(height):
            for c in range(width):
                format_data.append(allFeatTensor[:,r,c])
                ind_mapper[len(format_data)-1] = (r,c)

        format_data = np.array(format_data)
        format_data = imputeRowMin(format_data)
        format_data = np.log2(format_data)

        plt.figure()
        if dm_method == "PCA":
            pca = PCA(n_components=num_latent)
            format_data = pca.fit_transform(format_data)
            plt.xlabel("PC1 (" + str(np.round(100*pca.explained_variance_ratio_[0],2)) + "%)")
            plt.ylabel("PC2 (" + str(np.round(100*pca.explained_variance_ratio_[1],2)) + "%)")

        elif dm_method == "TSNE":
            tsne = TSNE(n_components=2)
            format_data = tsne.fit_transform(format_data)
            plt.xlabel("t-SNE1")
            plt.ylabel("t-SNE2")

        labels = kmean.fit_predict(format_data)
        group0Int = np.mean([sumImage[ind_mapper[x][0],ind_mapper[x][1]] for x in range(len(labels)) if labels[x] < .5])
        group1Int = np.mean([sumImage[ind_mapper[x][0],ind_mapper[x][1]] for x in range(len(labels)) if labels[x] > .5])

        if group0Int > group1Int:
            labels = labels < .5

        plt.scatter(format_data[:,0],format_data[:,1],c=labels)


        imageBoundary = np.zeros(sumImage.shape)
        for x in range(len(labels)):
            if labels[x] > .5:
                imageBoundary[ind_mapper[x][0],ind_mapper[x][1]] = 1

    imageBoundary = ndimage.binary_fill_holes(imageBoundary)

    return imageBoundary

def write_file_to_zip(myzip,filename):
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

def getImage(data,mz,nrows,ncols):
    # collect coordinate intensity pairs
    picDict = {}
    for index, row in data.iterrows():
        picDict[(index[2], index[1])] = row[mz]

    # covert to image coordinates
    xcords = [float(y) for y in list(set([x[0] for x in picDict]))]
    ycords = [float(y) for y in list(set([x[1] for x in picDict]))]

    # make output array and resize if necessary
    outarray = np.zeros((len(xcords), len(ycords)))

    xcords.sort()
    ycords.sort()
    xcordMap = {x: i for x, i in zip(xcords, range(len(xcords)))}
    ycordMap = {x: i for x, i in zip(ycords, range(len(ycords)))}

    # make output array
    for [x, y], intens in picDict.items():
        outarray[xcordMap[float(x)], ycordMap[float(y)]] = float(intens)
    #outarray = scale(outarray, nrows, ncols)

    return outarray

def normalizeTensor(tensorFilt):
    for r in range(len(tensorFilt[0])):
        for c in range(len(tensorFilt[0][0])):
            sumInt = np.sum(tensorFilt[:, r, c])
            tensorFilt[:, r, c] = tensorFilt[:, r, c] / max([1, sumInt])


def getISAEq(numCarbons):
    d = {16:palmitateISA,14:myristicISA,18:stearicISA,20:arachidonicISA}
    return d[numCarbons]

def K_means_image_split(tensor,k,dm_method="PCA",num_latent=2):
    # go through all features in dataset

    kmean = KMeans(k)
    format_data = []
    ind_mapper = {}
    for r in range(len(tensor[0])):
        for c in range(len(tensor[0][0])):
            format_data.append(tensor[:,r,c])
            ind_mapper[len(format_data)-1] = (r,c)

    format_data = np.array(format_data)
    format_data = imputeRowMin(format_data)
    format_data = np.log2(format_data)

    if dm_method == "PCA":
        plt.figure()
        pca = PCA(n_components=num_latent)
        format_data = pca.fit_transform(format_data)
        if num_latent > 1:
            plt.xlabel("PC1 (" + str(np.round(100*pca.explained_variance_ratio_[0],2)) + "%)")
            plt.ylabel("PC2 (" + str(np.round(100*pca.explained_variance_ratio_[1],2)) + "%)")

    elif dm_method == "TSNE":
        plt.figure()
        tsne = TSNE(n_components=2)
        format_data = tsne.fit_transform(format_data)
        plt.xlabel("t-SNE1")
        plt.ylabel("t-SNE2")

    labels = kmean.fit_predict(format_data)

    if num_latent > 1:
        plt.scatter(format_data[:,0],format_data[:,1],c=labels)


    imageLabels = np.zeros(tensor[0].shape)
    for x in range(len(labels)):
        imageLabels[ind_mapper[x][0],ind_mapper[x][1]] = labels[x]

    return imageLabels