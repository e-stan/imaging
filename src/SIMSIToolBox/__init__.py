#### packages to import ###
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
from bisect import bisect_left
from bisect import insort_left
import random as rd
from copy import deepcopy

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
        self.data_tensor = np.array([]) #structure to store data level 0 = ion, level 1 = rows, level 2 = columns
        self.ppm = ppm #mass tolerance
        self.polarity = 0 #polarity
        self.targets = targets #mzs of targets, corresponds to level 0 of the data tensor
        self.tic_image = -1 #total ion current image
        self.mass_range = mass_range
        self.imageBoundary = -1 #thresholding image
        self.numCores = numCores #number of processor cores to use
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
        self.mass_errors = np.zeros((len(self.targets),self.tic_image.shape[0],self.tic_image.shape[1]))

        for [x,y],[intensities,ppmErrs] in zip(inds,result):
            self.data_tensor[:,x,y] = intensities
            self.mass_errors[:,x,y] = ppmErrs

        self.imageBoundary = np.ones(self.tic_image.shape)

    def from_pandas(self,df,polarity):
        """
        Read in data from a pandas DataFrame (that was generated from the to_pandas method)
        :param df: DataFrame, df to read from
        :param polarity: str, polarity of data ("negative, "positive")
        :return: None
        """
        self.polarity = polarity
        targetsFound = [x for x in df.columns.values if x not in ["x","y","tic","boundary"]]
        xdim = len(set(df["x"].values))
        ydim = len(set(df["y"].values))

        mapper = {mz:[col for col in targetsFound if 1e6 * np.abs(float(col)-mz)/mz < self.ppm] for mz in self.targets}

        args = [[df[["x","y"] + mapper[x]],mapper[x],self.intensityCutoff,(ydim,xdim)] for x in self.targets]
        args.append([df[["x","y","tic"]],["tic"],0,(ydim,xdim)])
        args.append([df[["x","y","boundary"]],["boundary"],-1,(ydim,xdim)])
        res = np.array(startConcurrentTask(MSIData.parse_df_to_matrix,args,self.numCores,
                                                        "reading csv",len(args)))

        self.tic_image = res[-2]
        self.imageBoundary = res[-1]

        self.data_tensor = np.array(res[:-2])
        self.mass_errors = np.zeros(self.data_tensor.shape)


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


    def to_pandas(self):
        """
        reformat data as pandas df
        :return: DataFrame, pandas dataframe with MSI data
        """
        #get dimensions of data
        nrows = self.data_tensor.shape[1]
        ncols = self.data_tensor.shape[2]
        ntotal = nrows * ncols

        args = [[x] for x in self.data_tensor] + [[self.tic_image], [self.imageBoundary]]

        data = np.array(startConcurrentTask(MSIData.parse_matrix_to_column,args,self.numCores,"forming matrix",len(args))).transpose()

        df = pd.DataFrame(data,index=range(ntotal),columns = list(self.targets) + ["tic","boundary"])

        x = []
        y = []

        for r in range(nrows):
            for c in range(ncols):
                x.append(c)
                y.append(r)

        df["x"] = x
        df["y"] = y

        df = df[["x", "y"] + list(df.columns.values[:-2])]

        return df

    def segmentImage(self,method="TIC_auto", threshold=0, num_latent=2, dm_method="PCA",fill_holes = True,n=None):
        """
        Segment image into sample and background
        :param method: str, method for segmentation ("TIC auto" = find optimal separation between background and foreground based on TIC intensity, "K_means"=use K-means clustering, "TIC_manual"= use a user-defined threshold for segment image based on TIC
        :param threshold: float, intensity threshold for TIC_manual method
        :param num_latent: int, number of latent variables to use in dimensionality reduction prior to clustering when using K_means
        :param dm_method: str, dimensionality reduction method to use with K_means ("PCA" or "TSNE")
        :param fill_holes: bool, True or False depending
        :param n: int, number of pixels to use for fitting kmeans
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

            if type(n) == type(None):
                n = len(format_data)

            if n > len(format_data):
                n = len(format_data)

            samp = rd.sample(list(range(len(format_data))),k=n)

            kmean.fit(format_data[samp])

            labels = kmean.predict(format_data)

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
        """
        Smooth data
        :param method: str, smoothing method "MA", "GB", "iMA", or "iGB". "iMA" and "iGB" only apply the smoothing filter to missing values (0s)
        :param kernal_size: int, size of smoothing box (e.g.,3=3x3, 5=5x5)
        :return: None
        """
        # apply moving average filter
        offset = int((kernal_size - 1) / 2)
        height,width = self.tic_image.shape

        result = startConcurrentTask(convolveLayer,[[offset, height, width, self.data_tensor[t], self.imageBoundary, method] for t in
                                            range(len(self.targets))],self.numCores,"Smoothing data",len(self.targets))

        tensorFilt = np.array(result)

        tic_smoothed = convolveLayer(offset,height,width,self.tic_image,self.imageBoundary,method)


        self.data_tensor = tensorFilt
        self.tic_image = tic_smoothed

    def runISA(self,inds=None,isaModel="flexible",T=[0,0,1],X_image = None,minIso=2,minFrac=0.0,NACorrected=False):
        """
        Run isotopomer spectral analysis on data
        :param inds: indices of data_tensor that contain the isotopes of the fatty acid of interest in order
        :param isaModel: str, ISA model to use. "flexible" infers g(t) and X directly without D, "classical" infers both D and g(t), "dual" infers only g(t). With dual, X_image must be included
        :param T: list, input tracer abundance, must be provided for classical
        :param X_image: list, list of numpy matrices that give the precursor labeling of each isotope (M0, M1, M2), required for dual
        :param minIso: int, minimum number of detected isotopes at each pixel required for ISA of that pixel
        :param minFrac: float, minimum relative intensity of isotope to conisder as detected
        :param NACorrected: bool, True is data has been natural abundance corrected, False otherwise
        :return: g(t) image,D image,X0 image,X1 image,X2 image,list of precursor labels,list of observed product labels,
        list of fit product labels,list that gives the number of isotopes detected in each pixel,error in product fit,
        labeling of product where ISA failed
        """
        if inds == type(None):
            inds = list(range(len(self.data_tensor)))
        if NACorrected:
            c13ab = 0.0
        else:
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

        numCarbons = len(self.data_tensor[inds]) - 1
        data = normalizeTensor(self.data_tensor[inds])
        func = getISAEq(numCarbons)

        numFounds = []

        for r in range(self.tic_image.shape[0]):
            for c in range(self.tic_image.shape[1]):
                # get product labeling
                P = data[:, r, c]

                goodInd = [x for x in range(len(P)) if P[x] > minFrac]

                # if not on background pixel
                if self.imageBoundary[r, c] > .5 and len(goodInd) > minIso:
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

    def correctNaturalAbundance(self,formulas,inds):
        """
        corrected natural abundance isotope signals in data
        :param formulas: list, formulas of metabolites whose abundance should be corrected
        :param inds: list, indices of metabolites corresponding to each formula
        :return: None
        """

        if self.polarity == "positive": charge = 1
        else: charge = -1
        args = []
        indMap = []
        allCoords = []
        for formula,ind in zip(formulas,inds):
            coords = []
            vecs = []
            for r in range(self.tic_image.shape[0]):
                for c in range(self.tic_image.shape[1]):
                    if self.imageBoundary[r,c] > 0.5:
                        vec = self.data_tensor[ind,r,c]
                        vecs.append(vec)
                        #args.append([vec,formula,charge])
                        coords.append([r,c])

            coords = splitList(coords,self.numCores)
            vecs = splitList(vecs,self.numCores)
            indMap += [ind for _ in vecs]

            args += [[np.array(vec),formula,charge] for vec in vecs]
            allCoords += coords
        results = startConcurrentTask(correctNaturalAbundance,args,self.numCores,"correcting natural abundance",len(args))

        for vec,coord,ind in zip(results,allCoords,indMap):
            for corr,(x,y) in zip(vec,coord):
                self.data_tensor[ind,x,y] = corr

    @staticmethod
    def parse_df_to_matrix(df,cols,intensityCutoff,dims,q=None):
        arr = np.zeros(dims)
        for index,row in df.iterrows():
            i = np.sum(row[cols].values)
            if i > intensityCutoff:
                arr[int(row["y"]),int(row["x"])] = i

        if type(q) != type(None):
            q.put([])

        return arr

    @staticmethod
    def parse_matrix_to_column(arr,q=None):

        arr = arr.flatten()

        if type(q) != type(None):
            q.put([])

        return arr


################## helper functions #########################


def showImage(arr, cmap):
    plt.imshow(arr, cmap=cmap)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])


def saveTIF(arr, filename):
    im = Image.fromarray(arr)
    im.save(filename)


def write_file_to_zip(myzip, filename):
    myzip.write(filename,
                arcname=filename.split("/")[-1])


def splitList(l, n):
    n = int(np.ceil(len(l) / float(n)))
    return list([l[i:i + n] for i in range(0, len(l), n)])


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def startConcurrentTask(task, args, numCores, message, total, chunksize="none", verbose=True):
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


def updateProgress(q, total, message=""):
    counter = 0
    while counter != total:
        if not q.empty():
            q.get()
            counter += 1
            printProgressBar(counter, total, prefix=message, printEnd="")


######################## Spectra and matrix operations ##########################

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



def convolveLayer(offset, height, width, layer, imageBoundary, method="MA", q=None):
    # iterate through pixels
    if "MA" in method:
        tensorFilt = np.zeros((height - 2 * offset, width - 2 * offset))
        for r in range(offset, height - offset):
            for c in range(offset, width - offset):
                if method[0] != "i" or layer[r,c] < 1e-6:
                    tempMat = layer[r - offset:r + offset + 1, c - offset:c + offset + 1]
                    coef = imageBoundary[r - offset:r + offset + 1, c - offset:c + offset + 1]
                    coef = coef / max([1, np.sum(coef)])
                    tensorFilt[r - offset, c - offset] = np.sum(np.multiply(tempMat, coef))
                else:
                    tensorFilt[r - offset, c - offset] = layer[r,c]
    elif "GB" in method:

        tensorFilt = deepcopy(layer)

        if "i" == method[0]:
            zeros = layer < 1e-6
            tensorFilt[zeros] = ndimage.gaussian_filter(layer, offset)[zeros]
        else:
            tensorFilt = ndimage.gaussian_filter(layer, offset)

    if type(q) != type(None):
        q.put(0)

    return tensorFilt


def correctNaturalAbundance(vecs, formula, charge=-1, q=None):
    data = pd.DataFrame(data=vecs,
                        columns=["No label"] + [str(x + 1) + "C13" for x in range(vecs.shape[1] - 1)])
    vec_cor = picor.calc_isotopologue_correction(data, molecule_formula=formula, molecule_charge=charge,
                                                 resolution_correction=False).values  # ,resolution=resolution,mz_calibration=res_mz).values[0]

    if type(q) != type(None):
        q.put(0)

    return vec_cor


def convertSpectraAndExtractIntensity(mzs, inten, thresh, targets, ppm, dtype, q=None):
    intensities = []
    ppms = []

    order = list(range(len(mzs)))
    order.sort(key=lambda x: mzs[x])

    mzs = [mzs[x] for x in order]
    inten = [inten[x] for x in order]

    for mz in targets:
        width = ppm * mz / 1e6
        mz_start = mz - width
        mz_end = mz + width
        Origpos = bisect_left(mzs, mz)
        if Origpos == len(mzs):
            Origpos -= 1
        pos = int(Origpos)
        val = 0
        observedMzs = [[0, 0]]
        if mzs[Origpos] > mz_start and mzs[Origpos] < mz_end:
            while pos >= 0:
                if mzs[pos] < mz_start:
                    break
                if inten[pos] > thresh:
                    val += inten[pos]
                    observedMzs.append([mzs[pos], inten[pos]])
                pos -= 1

            pos = int(Origpos)
            while pos < len(mzs):
                if mzs[pos] > mz_end:
                    break
                if inten[pos] > thresh:
                    val += inten[pos]
                    observedMzs.append([mzs[pos], inten[pos]])

                pos += 1

        if len(observedMzs) > 1:
            observedMzs = np.array(observedMzs)
            observedMzs[:, 1] = observedMzs[:, 1] / val
            observedMz = np.sum([x[0] * x[1] for x in observedMzs])
            err = np.abs(observedMz - mz) / mz * 1e6
        else:
            err = 0
        ppms.append(err)
        intensities.append(val)

    if type(q) != type(None):
        q.put(0)

    return np.array(intensities), np.array(ppms)


def convertSpectraArraysToDict(mzs, inten, thresh):
    return {mz: i for mz, i in zip(mzs, inten) if i > thresh}


def mergeMzLists(old, new, ppm):
    old.sort()
    for x in new:
        if len(old) > 0:
            width = ppm * x / 1e6
            mi = x - width
            ma = x + width
            closest = take_closest(old, x)
            if closest < mi or closest > ma:
                insort_left(old, x)
        else:
            old.append(x)
    return old


def getMzsOfIsotopologues(formula, elementOfInterest="C"):
    # calculate relevant m/z's
    m0Mz = f = molmass.Formula(formula)  # create formula object
    m0Mz = f.isotope.mass  # get monoisotopcic mass for product ion
    # get number of carbons
    comp = f.composition()
    for row in comp:
        if row[0] == elementOfInterest:
            numCarbons = int(row[1])

    mzsOI = [m0Mz + 1.00336 * x for x in range(numCarbons + 1)]
    return m0Mz, mzsOI, numCarbons


def normalizeTensor(tensorFilt):
    normalizedTensor = np.zeros(tensorFilt.shape)
    for r in range(len(tensorFilt[0])):
        for c in range(len(tensorFilt[0][0])):
            sumInt = np.sum(tensorFilt[:, r, c])
            normalizedTensor[:, r, c] = tensorFilt[:, r, c] / max([1, sumInt])
    return normalizedTensor


####################### ISA functions and equations #######################

def getISAEq(numCarbons):
    d = {16: palmitateISA, 18: stearicISA, 20: arachidonicISA}
    return d[numCarbons]


def objectiveFunc(t, p, goodInd, params=[], alpha=0, lam=0):
    trel = np.array([t[x] for x in goodInd])
    prel = np.array([p[x] for x in goodInd])

    trel = trel / np.sum(trel)
    prel = prel / np.sum(prel)

    return np.sum(np.square(np.subtract(trel, prel))) + alpha * lam * np.sum(np.abs(params)) + (
            1 - alpha) / 2 * lam * np.sum(np.square(params))


def generalizedExp(t, c, k):
    return c * (1 - np.exp(-1 * k * t))


def ISAFit(T, N, P, func, goodInd, x_init=np.random.random((1)), plot=False, q=None):
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


def ISAFit_classical(T, N, P, func, goodInd, x_init=np.random.random((2)), plot=False, q=None):
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


def ISAFit_knownT(T, N, P, func, goodInd, x_init=np.random.random((1)), plot=False, q=None):
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

