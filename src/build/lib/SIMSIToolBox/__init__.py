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
import sympy as sym

class MSIData():
    def __init__(self,targets,ppm,mass_range = [0,1000],numCores = 1,intensityCutoff = 100,target_ccs=None,ccs_tol=None):
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
        self.target_ccs = target_ccs
        self.ccs_tol = ccs_tol


    def readimzML(self,filename,dims=None,include_mobility=False):
        """
        Load data from an imzml file with path filename.
        :param filename: str, path to imzml file
        :param dims: iterable, dimensions of image if imzml dimensions are corrupted. Optional
        :return: None
        """
        print("reading polarity...")
        p = ImzMLParser(filename,include_mobility=include_mobility)  # load data
        self.polarity = p.polarity
        print("done. Preparing to parse...")
        if type(dims) != type(None):
            p.imzmldict["max count of pixels x"] = dims[0]
            p.imzmldict["max count of pixels y"] = dims[1]

        whole_range_ppm = 1e6*(self.mass_range[1] - self.mass_range[0])/np.mean(self.mass_range)

        args = [[(filename,include_mobility,dims),np.mean(self.mass_range),None, whole_range_ppm,None,0]]

        if include_mobility:
            for mz,ccs in zip(self.targets,self.target_ccs):
                args.append([(filename,include_mobility,dims),mz,ccs,self.ppm,self.ccs_tol,self.intensityCutoff])
        else:
            for mz in self.targets:
                args.append([(filename,include_mobility,dims),mz,None,self.ppm,self.ccs_tol,self.intensityCutoff])



        # for idx, (x,y,_) in enumerate(p.coordinates):
        #     mzs, intensities = p.getspectrum(idx)
        #     args.append([mzs,intensities,self.intensityCutoff,self.targets,self.ppm,"centroid"])
        #     inds.append([y-1,x-1])

        print("starting parse")

        result = startConcurrentTask(convertSpectraAndExtractIntensity,args,self.numCores,"extracting intensities",len(args))

        self.tic_image = result[0]
        self.data_tensor = np.zeros((len(self.targets),self.tic_image.shape[0],self.tic_image.shape[1]))
        
        #self.mass_errors = np.zeros((len(self.targets),self.tic_image.shape[0],self.tic_image.shape[1]))
        result = result[1:]

        for i in range(len(result)):
            self.data_tensor[i] = result[i]

        # for [x,y],[intensities,ppmErrs] in zip(inds,result):
        #     self.data_tensor[:,x,y] = intensities
            #self.mass_errors[:,x,y] = ppmErrs

        self.imageBoundary = np.ones(self.tic_image.shape)

        print("done")

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
        fluxImageE = np.zeros(self.tic_image.shape)
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
        exprList = generateISAExpressions(getISAEq(numCarbons),numCarbons)

        numFounds = []

        if isaModel == "flexible":
            initParameters = [0.5]
        elif isaModel == "classical":
            initParameters = [0.5,0.5]
        elif isaModel == "elongation":
            initParameters = [0.33,0.33,0.33]
        else:
            print("ISA model not recognized")
            return -1

        for r in range(self.tic_image.shape[0]):
            for c in range(self.tic_image.shape[1]):
                # get product labeling
                P = data[:, r, c]

                goodInd = [x for x in range(len(P)) if P[x] > minFrac]

                # if not on background pixel
                if self.imageBoundary[r, c] > .5 and len(goodInd) > minIso:
                    # fit ISA
                    numFounds.append(len(goodInd))
                    a = [T, N, P, exprList, goodInd, np.array(initParameters)]
                    coords.append((r, c))
                    P_consider.append(P)
                    argList.append(a)  # np.random.random(1)))

        if isaModel == "flexible":
            eq = ISAFit
        elif isaModel == "classical":
            eq = ISAFit_classical
        elif isaModel == "elongation":
            eq = ISAFit_e_g

        results = startConcurrentTask(eq,argList,self.numCores,"Running ISA",len(argList))

        errors = []
        for (g, e,D, T_found, err, P_pred), (r, c), P_true in zip(results, coords, P_consider):
            # save results in data structures
            if g > -.0001 and g < 1.1 and all(xx > -0.01 and xx < 1.1 for xx in T_found) and e > -.0001 and e < 1.1 :
                errs.append(err)
                T_founds.append(T_found)
                P_preds.append(P_pred)
                P_trues.append(P_true)

                fluxImageG[r, c] = g
                fluxImageE[r, c] = e
                fluxImageD[r, c] = D
                fluxImageT0[r, c] = T_found[0]
                fluxImageT1[r, c] = T_found[1]
                fluxImageT2[r, c] = T_found[2]
            else:
                errors.append(P_true)

        return fluxImageG,fluxImageE,fluxImageD,fluxImageT0,fluxImageT1,fluxImageT2,T_founds,P_trues,P_preds,numFounds,errs,errors

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

def takeClosestInd(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return pos
    if pos == len(myList):
        return len(myList) - 1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos-1


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




def convertSpectraAndExtractIntensity(parser_info, mz,ccs, ppm,ccs_tol, intensity, q=None):
    filename,include_mobility,dims = parser_info
    p = ImzMLParser(filename,include_mobility=include_mobility)  # load data
    if type(dims) != type(None):
        p.imzmldict["max count of pixels x"] = dims[0]
        p.imzmldict["max count of pixels y"] = dims[1]

    width = ppm * mz / 1e6

    if ccs is None:
        image = getionimage(p, mz, width)
    else:
        image = getionimage(p,mz,width,ccs,ccs_tol)
    

    if type(q) != type(None):
        q.put(0)

    image[image < intensity] = 0.0

    return image


# def convertSpectraAndExtractIntensity(mzs, inten, thresh, targets, ppm, dtype, q=None):
#     intensities = []
#     ppms = []

#     order = list(range(len(mzs)))
#     order.sort(key=lambda x: mzs[x])

#     mzs = [mzs[x] for x in order]


#     inten = [inten[x] for x in order]

#     for mz in targets:
#         width = ppm * mz / 1e6
#         mz_start = mz - width
#         mz_end = mz + width

#         Origpos = takeClosestInd(mzs,mz)

#         pos = int(Origpos)
#         val = 0
#         observedMzs = [[0, 0]]

#         mzPath = []
#         if mzs[Origpos] > mz_start and mzs[Origpos] < mz_end:
#             while pos >= 0:
#                 mzPath.append(mzs[pos])
#                 if mzs[pos] < mz_start:
#                     break
#                 if inten[pos] > thresh:
#                     val += inten[pos]
#                     observedMzs.append([mzs[pos], inten[pos]])
#                 pos -= 1

#             pos = int(Origpos) + 1
#             while pos < len(mzs):
#                 mzPath.append(mzs[pos])
#                 if mzs[pos] > mz_end:
#                     break
#                 if inten[pos] > thresh:
#                     val += inten[pos]
#                     observedMzs.append([mzs[pos], inten[pos]])

#                 pos += 1

#         #if len(observedMzs)-1 != len([x for x in mzs if x > mz_start and x < mz_end]):
#         #    print(mzs[Origpos],mz_start,mz_end,mzPath,observedMzs,[x for x in mzs if x > mz_start and x < mz_end])

#         if len(observedMzs) > 1:
#             observedMzs = np.array(observedMzs)
#             observedMzs[:, 1] = observedMzs[:, 1] / val
#             observedMz = np.sum([x[0] * x[1] for x in observedMzs])
#             err = np.abs(observedMz - mz) / mz * 1e6
#         else:
#             err = 0
#         ppms.append(err)
#         intensities.append(val)

#     if type(q) != type(None):
#         q.put(0)

#     return np.array(intensities), np.array(ppms)



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
    f = molmass.Formula(formula)  # create formula object
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


def generateISAExpressions(expr,nc):
    symbols,expr = expr
    expr = sym.expand(expr)
    term_grouping = [0 for _ in range(nc+1)]
    symMapper = {"x1": 1, "x2": 2, "n1": 1, "n2": 2}

    for term in sym.Add.make_args(expr):
        isoCount = 0

        for val in sym.Mul.make_args(term):
            tmp = str(val)
            tmp = tmp.split("**")
            if tmp[0] in symMapper:
                n = symMapper[tmp[0]]
            else:
                n = 0
            if len(tmp) == 2:
                exp = int(tmp[1])
            else:
                exp = 1

            isoCount += int(n * exp)

        term_grouping[isoCount] += term

    #print(term_grouping)

    return symbols,term_grouping

def getISAEq(numCarbons):

    x0 = sym.Symbol('x0')
    x1 = sym.Symbol('x1')
    x2 = sym.Symbol('x2')
    n0 = sym.Symbol('n0')
    n1 = sym.Symbol('n1')
    n2 = sym.Symbol('n2')
    g = sym.Symbol('g')
    o = sym.Symbol("o")
    e = sym.Symbol('e')

    symbols = [x0,x1,x2,n0,n1,n2,g,o,e]

    if numCarbons == 16:
        expr = g * (x0 + x1 + x2) ** 8 + o * (n0 + n1 + n2) ** 8

    if numCarbons == 18:
        expr = g * (x0 + x1 + x2) ** 9 + e * (x0 + x1 + x2) * (n0 + n1 + n2) ** 8 + o * (n0 + n1 + n2) ** 9

    if numCarbons == 20:
        expr = e * (x0 + x1 + x2) * (n0 + n1 + n2) ** 9 + o * (n0 + n1 + n2) ** 10

    return symbols,expr


def objectiveFunc(t, p, goodInd, params=[], alpha=0, lam=0):
    trel = np.array([t[x] for x in goodInd])
    prel = np.array([p[x] for x in goodInd])

    trel = trel / np.sum(trel)
    prel = prel / np.sum(prel)

    return np.sum(np.square(np.subtract(trel, prel))) + alpha * lam * np.sum(np.abs(params)) + (
            1 - alpha) / 2 * lam * np.sum(np.square(params))

def ISAFit(T, N, P, exprList, goodInd, x_init, q=None):
    success = False
    initial_params = np.concatenate((x_init, T), axis=None)

    func = sym.lambdify(*exprList)

    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, func(x[1]/np.sum(x[1:4]),x[2]/np.sum(x[1:4]),x[3]/np.sum(x[1:4]),N[0],N[1],N[2],x[0],1-x[0],0), goodInd),
            #lambda x: objectiveFunc(P, func(x[0], 1.0, [xx / np.sum(x[1:]) for xx in x[1:]], N, P), goodInd),
            x0=initial_params)#,  # ,method="trust-constr",
        #bounds=[(0, x) for x in [1,np.inf,np.inf,np.inf]])
        if not sol.success:
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    # g, D = sol.x[:2]
    g = sol.x[0]
    o = 1-g
    e = 0
    T = sol.x[1:4]
    T = T / np.sum(T)
    D = 1.0
    err = sol.fun
    P_pred = func(T[0],T[1],T[2],N[0],N[1],N[2],g,o,e)
    P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0

    if type(q) != type(None):
        q.put(0)

    return g, e,D, T,err, P_pred

def ISAFit_e_g(T, N, P, exprList, goodInd, x_init, q=None):
    success = False
    initial_params = np.concatenate((x_init, T), axis=None)

    func = sym.lambdify(*exprList)

    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, func(x[3]/np.sum(x[3:6]),x[4]/np.sum(x[3:6]),x[5]/np.sum(x[3:6]),N[0],N[1],N[2],x[0]/np.sum(x[:3]),x[1]/np.sum(x[:3]),x[2]/np.sum(x[:3])), goodInd) + 0.001 * (x[0] + x[2])/np.sum(x[:3]),
            #lambda x: objectiveFunc(P, func(x[0], 1.0, [xx / np.sum(x[1:]) for xx in x[1:]], N, P), goodInd),
            x0=initial_params,  # ,method="trust-constr",
        bounds=[(0, x) for x in [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]])
        if not sol.success:
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True
    # g, D = sol.x[:2]
    g = sol.x[0] / np.sum(sol.x[:3])
    o = sol.x[1] / np.sum(sol.x[:3])
    e = sol.x[2] / np.sum(sol.x[:3])
    T = sol.x[3:6]
    T = T / np.sum(T)
    D = 1.0
    err = sol.fun
    P_pred = func(T[0],T[1],T[2],N[0],N[1],N[2],g,o,e)
    P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0

    if type(q) != type(None):
        q.put(0)

    return g, e,D, T,err, P_pred


def ISAFit_classical(T, N, P, exprList, goodInd, x_init, q=None):
    success = False
    initial_params = x_init

    func = sym.lambdify(*exprList)

    while not success:
        sol = opt.minimize(
            lambda x: objectiveFunc(P, func(T[0],T[1],T[2],N[0],N[1],N[2], x[0],1-x[0],0), goodInd),
            x0=initial_params,
        )  # bounds=[(0, 1) for _ in range(len(x_init) + len(T))])
        if not sol.success:
            initial_params = np.random.random(initial_params.shape)
        else:
            success = True

    g = sol.x[0]
    D = sol.x[1]
    e = 0
    X = np.array(T)*D + (1-D) * np.array(N)

    err = sol.fun
    P_pred = func(X[0],X[1],X[2],N[0],N[1],N[2], g,1-g,0)

    P_pred = P_pred / np.sum(np.array(P_pred)[goodInd])
    for x in range(len(P_pred)):
        if x not in goodInd:
            P_pred[x] = 0

    if type(q) != type(None):
        q.put(0)

    return g, e, D, X, err, P_pred


