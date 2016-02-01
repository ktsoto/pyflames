from glob import glob
import os
import sys
import numpy as np
import pdb
from astropy.io import fits as pyfits
#import pyfits
#from astropy.stats.funcs import sigma_clip
from subprocess import call
from scipy import ndimage as ndi

###########################################################################################################################
#### The process  #########################################################################################################
###########################################################################################################################

#1 - Run pipeline in gasgano up to the standard star reduction
#2 - run pyflames.coslist() on a list of raw science files
#3 - Run pipeline giscience on the science frames with *_pp.fits
#4 - run pyflames.runsslist() on a list of 
#5 - run pyflames.fluxcalibrate()
#6 - run pyflames.

###########################################################################################################################
#### cosmic ray removal ###################################################################################################
###########################################################################################################################

def bias(dfn,bfn):
    hd = pyfits.open(dfn)
    hb = pyfits.open(bfn)
    hd[0].data -= hb[0].data
    hd.writeto(dfn.replace('.fits','_bs.fits'))

def pycosmic(scifn):
    strcall = 'PyCosmic {0} {1} {2} 4.30 --fwhm 2 --rlim 1 --iter 6 --parallel'.format(scifn,
                                                                                       scifn.replace('.fits', '_msk.fits'),
                                                                                       scifn.replace('.fits', '_cln.fits'))
    print 'Calling PyCosmic'
    print strcall
    os.system(strcall)

def mergecos(scifn):
    hs = pyfits.open(scifn)
    hb = pyfits.open('master_bias_0000.fits')
    hcos = pyfits.open(scifn.replace('.fits','_bs_msk.fits'))
    hs[0].data = hcos[0].data + hb[0].data
    hs.writeto(scifn.replace('.fits','_pp.fits'))


def runcoslist(flst):
    for fl in flst:
        bias(fl, 'master_bias_0000.fits')
        pycosmic(fl.replace('.fits','_bs.fits'))
        mergecos(fl)

###########################################################################################################################
#### sky subtraction ######################################################################################################
###########################################################################################################################
        
def makesky(fn):
    h = pyfits.open(fn)
    srt = np.argsort(np.median(h[0].data, axis=0))
    skystack = h[0].data[:,srt[0:14]] #14 sky fibers
    sky = sigma_clip(skystack, axis=1, cenfunc=np.median).mean(axis=1)
    skyhdu = pyfits.PrimaryHDU(np.array(sky))
    newfn = fn.replace('.fits','_sky.fits')
    skyhdu.writeto(newfn)

def skysub(cfn,sfn):
    hc = pyfits.open(cfn)
    hs = pyfits.open(sfn)
    hc[0].data -= hs[0].data[:,np.newaxis,np.newaxis]
    hc.writeto(cfn.replace('.fits', '_ss.fits'))

def runsslist(flst):
    for fl in flst:
        makesky(fl.replace('cube_spectra','rbnspectra'))
        skysub(fl,fl.replace('cube_spectra','rbnspectra').replace('.fits','_sky.fits'))


###########################################################################################################################
##### flux calibrate ######################################################################################################
###########################################################################################################################

def fc(flst):
    for fl in flst:
        hir = pyfits.open(fl.split('/')[0]+'/instrument_response_0001.fits')
        std = hir[0].data.flatten()
        w = np.where(std == 0)
        std[std == 0] = np.median(std[np.min(w)-10:np.min(w)-5])
        sstd = ndi.uniform_filter(ndi.median_filter(std,500,mode='nearest'),500)
        hc = pyfits.open(fl)
        hc[0].data = hc[0].data/sstd[:,np.newaxis,np.newaxis]
        hc.writeto(fl.replace('_ss.fits','_fc.fits'))
        hir.close()
        hc.close()

def fcnoise(flst):
    for fl in flst:
        hir = pyfits.open(fl.split('/')[0]+'/instrument_response_0001.fits')
        std = hir[0].data.flatten()
        w = np.where(std == 0)
        std[std == 0] = np.median(std[np.min(w)-10:np.min(w)-5])
        sstd = ndi.uniform_filter(ndi.median_filter(std,500,mode='nearest'),500)
        hc = pyfits.open(fl)
        hc[0].data = hc[0].data/sstd[:,np.newaxis,np.newaxis]
        hc.writeto(fl.replace('_ss.fits','_fc.fits'))
        hir.close()
        hc.close()


###########################################################################################################################
##### coadd ###############################################################################################################
###########################################################################################################################

def spaxmask(cube):
    bady = np.array([0, 0,13,13,  0,  0, 13, 13,  8, 9])
    badx = np.array([0, 1, 0, 1, 20, 21, 20, 21, 20, 20])  
    msk = np.zeros([14,22],dtype=bool)
    msk[(bady,badx)] = True
    mcube = np.ma.array(cube, mask=np.zeros(cube.shape, dtype=bool) + msk[np.newaxis,...])
    return mcube

def cubemerge(offsetfile):
    
    f = open(offsetfile)
    fn = []; x = []; y = []
    for line in f:
        sline = line.split()
        fn.append(sline[0])
        y.append(int(sline[1]))
        x.append(int(sline[2]))

    #This knows about argus' dimensions
    stack = np.ma.array(np.zeros([3730,14+int(max(y)-min(y)),22+int(max(x)-min(x)),len(fn)]),
                        mask = np.ones([3730,14+int(max(y)-min(y)),22+int(max(x)-min(x)),len(fn)]))

    refx = max(x)
    refy = max(y)

    for i in range(len(fn)):
        h = pyfits.open(fn[i])
        cube = h[0].data
        mcube=spaxmask(cube)
        xdif = refx - x[i]
        ydif = refy - y[i]
        stack[:,ydif:cube.shape[1]+ydif, xdif:cube.shape[2]+xdif, i] = mcube

    h = pyfits.open(fn[0])
    cube = np.array(stack.mean(axis=3))
    h[0].data = cube
    h[0].header['CRPIX1'] = h[0].header['CRPIX1'] + refx - x[0]
    h[0].header['CRPIX2'] = h[0].header['CRPIX2'] + refy - y[0]
    
    h.writeto(os.path.dirname(fn[0])[:-2]+'/combined_science_cube_spectra.fits')
    
def noisemerge(offsetfile):
    
    f = open(offsetfile)
    fn = []; x = []; y = []
    for line in f:
        sline = line.split()
        fn.append(sline[0])
        y.append(int(sline[1]))
        x.append(int(sline[2]))

    #This knows about argus' dimensions
    stack = np.ma.array(np.zeros([3730,14+int(max(y)-min(y)),22+int(max(x)-min(x)),len(fn)]),
                        mask = np.ones([3730,14+int(max(y)-min(y)),22+int(max(x)-min(x)),len(fn)]))

    refx = max(x)
    refy = max(y)

    for i in range(len(fn)):
        h = pyfits.open(fn[i])
        cube = h[0].data
        mcube=spaxmask(cube)
        xdif = refx - x[i]
        ydif = refy - y[i]
        stack[:,ydif:cube.shape[1]+ydif, xdif:cube.shape[2]+xdif, i] = mcube

    h = pyfits.open(fn[0])
    cube = np.array(stack.var(axis=3))
    h[0].data = cube
    h[0].header['CRPIX1'] = h[0].header['CRPIX1'] + refx - x[0]
    h[0].header['CRPIX2'] = h[0].header['CRPIX2'] + refy - y[0]
    
    h.writeto(os.path.dirname(fn[0])[:-2]+'/combined_var_cube_spectra.fits')
    
def appendvar(scifile, varfile):

    shdu = pyfits.open(scifile)
    vhdu = pyfits.open(varfile)

    nhdu = pyfits.HDUList(shdu[0])
    nhdu.append(vhdu[0])
    
    nhdu.writeto('sci_var_combined_cube_spectra.fits')
