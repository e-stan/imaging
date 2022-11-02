# code adpated from https://github.com/LaRoccaRaphael/MSI_recalibration

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from sklearn import linear_model
from scipy.stats import gaussian_kde
from copy import deepcopy
from multiprocessing import Pool,Manager
from threading import Thread
import random as rd
import matplotlib.pyplot as plt

def write_corrected_msi(msi,output_file,tolerance,database_exactmass,step,dalim,numCores):
    """
    perform adaptive pixel normalization
    :param msi: str, path to input imzml data file
    :param output_file: str, path for recalibrated imzml
    :param tolerance: float, maximum expected mass shift
    :param database_exactmass: list, sorted list of m/z values for calibrating compounds
    :param step: float, parameter for controlling smoothing of histogram. Higher = more smoothing, lower = less smoothing
    :param dalim: float, maximum expected spread of mass shifts in m/z
    :param numCores: int, number of processor cores to use
    :return: None
    """
    # iterate throug each pixel of an MSI
    print("reading spectra...",end="")
    args = []
    p = ImzMLParser(msi, parse_lib='ElementTree')
    coords = []
    for idx, (x, y, z) in enumerate(p.coordinates):

        ms_mzs, ms_intensities = p.getspectrum(idx)
        args.append([ms_mzs,ms_intensities,tolerance,database_exactmass,step,dalim])
        coords.append((x,y,z))

    print("done")

    result = startConcurrentTask(correctSpectrum,args,numCores,"correcting spectra",len(args))

    successCount = 0
    totalCount = 0
    with ImzMLWriter(output_file) as w:
        for (corrected_mzs,ms_intensities,success),(x,y,z) in zip(result,coords):
            w.addSpectrum(corrected_mzs, ms_intensities, (x,y,z))
            successCount += success
            totalCount += 1

    print("corrected " + str(100*float(successCount)/totalCount) + "% of pixels")
    print("writing spectra...",end="")
    print("done")

def visualizeParameters(msi,n,tolerance,database_exactmass,step,dalim):
    """
    visualize correction results for given parameters
    :param msi: str, path to input imzml data file
    :param n: number of pixels to show correction plots for
    :param tolerance: float, maximum expected mass shift
    :param database_exactmass: list, sorted list of m/z values for calibrating compounds
    :param step: float, parameter for controlling smoothing of histogram. Higher = more smoothing, lower = less smoothing
    :param dalim: float, maximum expected spread of mass shifts in m/z
    :return: None
    """
    p = ImzMLParser(msi, parse_lib='ElementTree')
    numPixels = list(range(len(p.coordinates)))
    samp = rd.sample(numPixels,k=n)
    i = 0
    for x in samp:
        plt.figure()
        plt.title(i)
        ms_mzs, ms_intensities = p.getspectrum(x)
        peaks_ind = peak_selection(ms_intensities)
        peaks_mz = ms_mzs[peaks_ind]

        print(len(peaks_mz), "peaks found")
        hit_exp, hit_errors = hits_generation(peaks_mz, database_exactmass, tolerance)

        x = np.asarray(hit_errors)
        x_grid = np.arange(-tolerance, tolerance + 0.0001, 0.0001)
        pdf = kde_scipy(x, x_grid, bandwidth=step)
        max_da_value = x_grid[np.argmax(pdf, axis=0)]
        plt.plot(x_grid,pdf)

        plt.plot([max_da_value - dalim,max_da_value - dalim],[0,max(pdf)],color="red")
        plt.plot([max_da_value + dalim,max_da_value + dalim],[0,max(pdf)],color="red")

        print(len(hit_errors),"hits found")
        roi = hits_selection(hit_errors, step, tolerance, da_limit=dalim)

        print(len(roi),"peaks in roi found")

        mz_error_model = create_lm(hit_exp, hit_errors, tolerance=tolerance, da_limit=dalim, step=step)

        plt.figure()
        plt.scatter(hit_exp,hit_errors,color="black",s=3)
        plt.plot([min(hit_exp),max(hit_exp)],[max_da_value + dalim,max_da_value + dalim],color="red")
        plt.plot([min(hit_exp),max(hit_exp)],[max_da_value - dalim,max_da_value - dalim],color="red")

        mzsToPlot = np.linspace(min(hit_exp),max(hit_exp),100)
        X = np.vander(mzsToPlot, 2)
        predicted_mz_errors = mz_error_model.predict(X)

        plt.plot(mzsToPlot,predicted_mz_errors,color="grey",alpha=0.5)
        plt.title(i)
        i += 1

##### HELPER FUNCTIONS #####

def peak_selection(ms_intensities):
    # return the 300 mot intense centroid of a mass spectrum
    intensities_arr = np.array(ms_intensities)
    return(intensities_arr.argsort()[::-1][:300])

def compute_masserror(experimental_mass, database_mass, tolerance):
    # mass error in Dalton
    if database_mass != 0:
        return abs(experimental_mass - database_mass) <= tolerance
       
def binarySearch_tol(arr, l, r, x, tolerance): 
    # binary with a tolerance in Da search from an ordered list 
    while l <= r: 
        mid = l + (r - l)//2; 
        if compute_masserror(x,arr[mid],tolerance): 
            itpos = mid +1
            itneg = mid -1
            index = []
            index.append(mid)
            if( itpos < len(arr)):
                while compute_masserror(x,arr[itpos],tolerance) and itpos < len(arr):
                    index.append(itpos)
                    itpos += 1 
            if( itneg > 0): 
                while compute_masserror(x,arr[itneg],tolerance) and itneg > 0:
                    index.append(itneg)
                    itneg -= 1     
            return index 
        elif arr[mid] < x: 
            l = mid + 1
        else: 
            r = mid - 1
    return -1

def hits_generation(peaks_mz,database_exactmass, tolerance): 
    # for each detected mz return its index in of the hits in the database
    hit_errors = list()
    hit_exp = list()
    for i in range(0,np.size(peaks_mz,0)):
        exp_peak = peaks_mz[i]
        db_ind = binarySearch_tol(np.append(database_exactmass,np.max(database_exactmass)+1),
                                  0, len(database_exactmass)-1, exp_peak,tolerance)
        if db_ind != -1:
            for j in range(0,len(db_ind)):
                true_peak = database_exactmass[db_ind[j]]
                da_error = (exp_peak - true_peak)
                hit_errors.append(da_error)
                hit_exp.append(exp_peak)
    return(np.asarray(hit_exp),np.asarray(hit_errors))


def kde_scipy(x, x_grid, bandwidth=0.002, **kwargs):
    # kernel density estimation of the hit errors 
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def hits_selection(hit_errors, step, tolerance, da_limit):
    # return the indexes of the hits of the most populated error region 
    x = np.asarray(hit_errors)
    x_grid = np.arange(-tolerance,tolerance+0.0001,0.0001)
    pdf = kde_scipy(x, x_grid, bandwidth=step)
    max_da_value = x_grid[np.argmax(pdf,axis=0)]
    roi = (x <= (max_da_value + da_limit)) & (x >= (max_da_value -da_limit ))
    return(roi) 


def create_lm(hit_exp,hit_errors,tolerance=30,da_limit=2.5,step=0.001):
    # estimate a linear model of the mz error according to the mz with RANSAC algorithm
    X = np.vander(hit_exp, 2) # 2d array for ransac algorithm, we add only ones in the second column 
    roi = hits_selection(hit_errors,step,tolerance=tolerance,da_limit=da_limit)
    y = hit_errors[roi]
    X = X[roi,]
    try:
        model = linear_model.RANSACRegressor(max_trials=300, min_samples=10)
        mz_error_model = model.fit(X, y)
    except ValueError:
        print("error")
        mz_error_model = []
    return(mz_error_model)

def correct_mz_lm(ms_mzs,mz_error_model):
    # predict the Da errors for each detected mz and correct them
    X = np.vander(ms_mzs, 2)
    predicted_mz_errors = mz_error_model.predict(X)
    estimated_mz = ms_mzs - predicted_mz_errors
    return(estimated_mz)

def correctSpectrum(ms_mzs,ms_intensities,tolerance,database_exactmass,step,dalim,q=None):
    peaks_ind = peak_selection(ms_intensities)
    peaks_mz = ms_mzs[peaks_ind]

    corrected_mzs = deepcopy(ms_mzs)
    success = False
    if len(peaks_mz) > 30:
        hit_exp, hit_errors = hits_generation(peaks_mz, database_exactmass, tolerance)
        if len(hit_errors) > 10:
            roi = hits_selection(hit_errors, step, tolerance, da_limit=dalim)
            if np.sum(roi) > 10:
                mz_error_model = create_lm(hit_exp, hit_errors, tolerance=tolerance, da_limit=dalim, step=step)
                if mz_error_model:
                    corrected_mzs = correct_mz_lm(ms_mzs, mz_error_model)
                    success = True
    if type(q) != type(None):
        q.put([])
    return corrected_mzs, ms_intensities,success


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

