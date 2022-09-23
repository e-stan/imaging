import SIMSIToolBox
import os
import matplotlib.pyplot as plt


inpath = "X:/MSI_Shared_Data/13CImagingManuscript/raw_data/imzmls/"
#inpath = "X:/Kevin/Bruker/MALDI/imzML/"
outpath = "X:/MSI_Shared_Data/13CImagingManuscript/raw_data/imzmls/subset_data/"
#outpath = inpath + "processed_data/"
metabolites = ["C6H8NO5","C6H12O9P","C16H31O2","C20H31O2","C5H8NO4"]

ppmThresh = 15
num_cores = 20
intensityCutoff = 0
convSquare = 1 #size of filter (1=1x1,3=3x3,5=5x5)
colormap = "gray" #coloring for images, see https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
dm_method = "PCA" #method for dimensionality reduction ("PCA" or "TSNE") PCA has worked better for me
seg_method = "K_means" #thresholding method ("TIC_auto", "K_means", "TIC_manual")
num_components = 2 #number of compoents to use with PCA or TSNE
filt = "GB" #filtering method (GB = gaussian blur, MA = moving average)


if __name__ == "__main__":

    mzs = []

    for met in metabolites:
        m0Mz, mzsOI, numCarbons = SIMSIToolBox.getMzsOfIsotopologues(met, elementOfInterest="C")
        mzs += list(mzsOI)

    files = [x for x in os.listdir(inpath) if ".imzML" in x or ".imzml" in x]
    for file in files:
        if file.replace(".imzml",".csv").replace(".imzML",".csv") not in os.listdir(outpath):
            print(file)
            dim = file.split("_")[2]
            dim = (int(dim.split("w")[0]),int(dim.split("w")[1][:-1]))
            msi = SIMSIToolBox.MSIData(mzs,ppm=ppmThresh,numCores = num_cores,intensityCutoff=intensityCutoff)
            #msi.readimzML(inpath+file)
            msi.readimzML(inpath+file, dim)

            msi.segmentImage(method=seg_method)
            plt.figure()
            SIMSIToolBox.showImage(msi.imageBoundary, cmap=colormap)
            msi.smoothData(filt,convSquare)
            df = msi.to_pandas()
            df.to_csv(outpath+file.replace(".imzml",".csv"))

    plt.show()
