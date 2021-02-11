# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
#from PySide import QtGui
from scipy.ndimage import imread
import ReadDMFile as dm3
from libtiff import TIFF

import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.plotly as py
import math
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from mpl_toolkits.mplot3d import Axes3D
import pickle
np.seterr(divide='ignore', invalid='ignore')
import sys
import os

def save_tiff(image, file_path, label):
    dir = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    base = os.path.splitext(base)[0]
    filename = os.path.join(dir, base+'_'+str(label)+'.tif')
    tif = TIFF.open(filename, mode='w')

    tif.write_image(image)

def save_data(x, y, file_path, label):
    dir = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    base = os.path.splitext(base)[0]
    filename = os.path.join(dir, base+'_'+str(label)+'.txt')
    fout = open(filename, 'w')
    fout_text = ''
    for i in range(len(y)):
        fout_text += str(x[i]) + ' ' + str(y[i])+'\n'
    fout.write(fout_text)
    fout.close()

def conv2(a, b):
    #fftconvolve returns the same value as in matlab in full mode, but for 'same' mode, it selects center differently. This method deals with the difference.
    c = fftconvolve(a,b,'full')
    n = c.shape
    m = a.shape
    dx = n[0]-m[0]
    dy = n[1]-m[1]

    endx = n[0]-math.floor(dx/2)
    startx = endx -m[0]


    endy = n[1]-math.floor(dy/2)
    starty = endy -m[1]
    return c[startx:endx, starty:endy]

def conv4(a, b):
    #fftconvolve returns the same value as in matlab in full mode, but for 'same' mode, it selects center differently. This method deals with the difference.
    c = fftconvolve(a,b,'full')
    n = c.shape
    m = a.shape
    dx = n[0]-m[0]
    dy = n[1]-m[1]
    dz = n[2]-m[2]
    dt = n[3]-m[3]    

    endx = n[0]-math.floor(dx/2)
    startx = endx -m[0]

    endy = n[1]-math.floor(dy/2)
    starty = endy -m[1]
    
    endz = n[2]-math.floor(dz/2)
    startz = endz -m[2]

    endt = n[3]-math.floor(dt/2)
    startt = endt -m[3]
    return c[startx:endx, starty:endy, startz:endz, startt:endt]        
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return math.sqrt(variance)

def run_me(file_path):


    print('Starting Processing the stacked images')

    # Parameter
    ovsam = 4              # oversampling rate
    rmax = 7               # search radius for covarianze (<half image size)
    mix = 0.9              # mixing factor for subpixel localization
    convkrit = 5           # criterion for localization of subpixelposition
    covr = []
    ded_flag = 0           # low intensity flag (single electron counting mode)


    #Slowscan
    kadim = [2048,2048]
    g = 18.5
    pixel = 14
    #NSFr = loadmat("C:/Users/lucaz/Documents/Bachelor_Arbeit/MATLAB/NSFr.mat")
    #covr = NSFr['covr']

    # Gitter bereitstellen
    print('Creating grid')
    temp = np.arange(-kadim[1]/2,(kadim[1]*ovsam/2)/ovsam, 1/float(ovsam))
    [X2,Y2] = np.meshgrid(temp, temp)

    [temp,R2] = cart2pol(X2,Y2)            # polar grid for radially symmetric MTF
    temp1 = np.arange(-rmax,rmax+1,1)
    [X3,Y3] = np.meshgrid(temp1,temp1)
    temp2 = np.arange(-rmax*ovsam-1, rmax*ovsam+1, 1)
    [X4,Y4] = np.meshgrid(temp2,temp2)

    # Bild laden

    #file_path, _ = QtGui.QFileDialog.getOpenFileName(None, 'Please choose an image file','.')

    #file_path = "C:/Users/Luca Zangari/Documents/Uni/USB backup/TEM Messung 1 05.02.2016/DM3/TEST/test.dm3"

    dx = 1
    dy = 1
    dz = 1
    if ('.em' in file_path): 
        expim = imread(file_path)
        tempx = np.array(expim)
        expim = tempx.astype(np.double);
    elif ('.dm3' in file_path):
        dm3f = dm3.DM3(file_path)
        tempx = np.array(dm3f.imagedata)
        expim = tempx.astype(np.double);
        dx = int(dm3f.imagexdim[0])
        dy = int(dm3f.imageydim[0])
        dz = int(dm3f.imagezdim[0])
    elif ('.tif' in file_path):
        tif = TIFF.open(file_path, 'r')

        dz = 0
        tempx = np.array([])
        for image in tif.iter_images():
            tempx = np.concatenate((tempx, np.ravel(image)))
            dz +=1
        dx = dy = np.sqrt(len(tempx)/dz)
        expim = tempx.astype(np.double);

    # Einzelelektronen herauslesen
    expim = np.reshape(expim, (dx, dy, dz), order="F")
    if (ded_flag == 1): #implement later
        agp = [32,32]
        maxVal = 100
        nn = 0
        temp = []
        temp2 = np.zeros(shape = expim.shape)
        temp2[agp[0]/2:agp[0]-agp[0]/2,agp[1]/2:agp[1]-agp[1]/2] = expim[agp[0]/2:agp[0]-agp[0]/2,agp[1]/2:agp[1]-agp[1]/2]

        while maxVal>20 and nn<=600:
            print(nn)
            maxVal = np.max(temp2.flatten())
            I = np.argmax(temp2.flatten())
            #[row, col] = ind2sub(kadim, I);
            row = I/temp2.shape[0]
            col = I%temp2.shape[0]

            temp[:,:,nn] = temp2[row-agp[0]/2-1:row+agp[0]/2-1,col-agp[1]/2-1:col+agp[1]/2-1]

            temp2[row-agp[0]/2-1:row+agp[0]/2-1,col-agp[1]/2-1:col+agp[1]/2-1] = 0
            nn = nn+1;

        expim = np.copy(temp)
        agp = expim.shape      

    agp = expim.shape
    agp2 =agp * ovsam

    # Dunkelstrom entfernen

    dark = np.copy(expim)
    dark[1:agp[0]-1,1:agp[1]-1,:] = 0

    temp = np.sum(dark, axis = 0)
    temp1 = np.sum(temp, axis =0)

    dark = np.squeeze(temp1/(2*(agp[0]+agp[1]-2)))

    for n in xrange(agp[2]):
        print('Clearing dark current')
        print(n)
        expim[:,:,n] = expim[:,:,n]-dark[n]

    # Belichtung normalisieren + Bilder entfernen (can be omitted)
    print('Integration...')
    temp = np.sum(expim, axis = 0)
    temp1 = np.sum(temp, axis =0)
    print('intefrated image intensity')
    Iin = np.squeeze(temp1) # integrated image intensity

    avIin = Iin.mean(axis =0)               # global average over whole series
    print('Global average over whole series')
    sIin = Iin.std(axis =0, ddof=1)                 # global standard deviation over whole series
    temp = np.zeros((agp[0], agp[1], agp[2]))

    m = 0
    tresh = 0.4
    for n in xrange(agp[2]):
        if (abs(Iin[n]-avIin) <= tresh*avIin):
            temp[:,:,m] = expim[:,:,n]
            m = m+1

    print ("m=" + str(m))
    expim = temp[:,:,0:m]

    agp = expim.shape
    print (agp)

    Iin = np.sum(np.sum(expim, axis = 0),axis = 0)

    avIin = np.squeeze(Iin).mean(axis =0)

    sIin = np.squeeze(Iin).std(axis =0)     

    if ded_flag==1:
        g = avIin

    # gleitende Mittelwerte gleichsetzen
    im = np.copy(expim);    

    # Gitter anlegen    
    agp = np.array(expim.shape)
    agp2 = agp * ovsam

    x = np.arange(-agp[0]/2,agp[1]/2,1)
    x2 = np.arange(-agp[0]/2, agp[1]/2, 1/float(ovsam))
    [X,Y] = np.meshgrid(x2,x2)
    [temp,R] = cart2pol(X,Y)                     

    #vorl.Statistik bestimmen, (normieren)

    imav = im.mean(axis = 2)

    imstd = im.std(axis = 2)

    immax = np.max(np.max(im, axis = 0), axis = 0)
    #
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1_y = np.squeeze(Iin)
    ax1.plot(ax1_y)
    save_data(xrange(len(ax1_y)), ax1_y, file_path, 'total_illumination')

    ax1.set_title('Total Illumination')

    ax2 = fig.add_subplot(gs[0,1])
    ax2_y = np.squeeze(immax)
    ax2.plot(ax2_y)
    save_data(xrange(len(ax2_y)), ax2_y, file_path, 'image_maximal_intensity')
    ax2.set_title('Image maximal intensity')
    #
    #
    #plt.tight_layout()
    #fig = plt.gcf()
    #plt.show()

    #Einzelaufnahmen ausrichten
    # oversampling Fourierraum

    temp = np.fft.fft2(imav)
    temp1 = np.fft.fftshift(temp)
    pw1 = (ovsam-1)*agp[0]/2
    pw2 = (ovsam-1)*agp[1]/2


    nan_pad = np.array(((0,0), (0,0)))

    temp2 = np.pad(temp1,((pw1,pw2),(pw1,pw2)),mode='constant', constant_values =nan_pad)
    temp3 = np.fft.ifftshift(temp2)
    temp4 = np.fft.ifft2(temp3)
    imav2 = ovsam*ovsam*(temp4.real)

    nan_pad_2 = np.array(((0,0), (0,0), (0,0)))
    temp = np.fft.fft2(im, axes = (0,1))
    temp1 = np.fft.fftshift(temp, axes = (0,1))
    temp2 = np.pad(temp1,((pw1,pw2),(pw1,pw2),(0,0)),mode='constant', constant_values =nan_pad_2)
    temp3 = np.fft.ifftshift(temp2, axes = (0,1))
    temp4 = np.fft.ifft2(temp3,axes = (0,1))
    im2 = ovsam*ovsam*(temp4.real)


    im4 = np.copy(im2)

    # oversampling Ortsraum
    print('Oversampling')
    tempim = im.shape

    im3 = np.zeros(shape=(tempim[0],tempim[1],tempim[2],ovsam,ovsam))

    im3[:,:,:,math.floor(ovsam/2),math.floor(ovsam/2)] = np.copy(im)

    im3 = np.reshape(np.transpose(im3,axes=[3,0,4,1,2]),[ovsam*tempim[0],ovsam*tempim[1],tempim[2]], order="F")

    temp = np.ones(shape=(tempim[0],tempim[1],tempim[2]))
    Nf = np.zeros(shape=(tempim[0],tempim[1],tempim[2],ovsam,ovsam))   # counts how many samples located on subpixel
    Nf[:,:,:,math.floor(ovsam/2),math.floor(ovsam/2)] = np.copy(temp)
    Nf = np.reshape(np.transpose(Nf,axes=[3,0,4,1,2]),[ovsam*tempim[0],ovsam*tempim[1],tempim[2]], order="F")

    xn = np.ones(agp[2])
    yn = np.copy(xn)
    m = 0
    xn2 = xn+1
    yn2 = yn+1



    while (agp[2]-np.sum(xn==xn2)> convkrit) or (agp[2]-np.sum(yn==yn2)>convkrit):
        m = m+1  
        print('Kovarianz')
        xn2 = np.copy(xn)
        yn2 = np.copy(yn)

        maxin = np.argmax(imav2.flatten(1))

        Iy = maxin/(imav2.shape[1])
        Ix = maxin % (imav2.shape[1])

        imav2 = np.roll((np.roll(imav2,agp[0]/2-Ix,axis=0)),agp[1]/2-Iy, axis=1)
        print('Displacement')
        for n in xrange(agp[2]):
            print(n)

            # Fitmethode
            tempim2 = im2[:,:,n]
            psfcov = conv2(tempim2,imav2)        
            IND = np.argmax(psfcov.flatten(1))

            I2 = IND/tempim2.shape[1]
            I1 = IND%tempim2.shape[1]

            # Verschieben        
            im4[:,:,n] = np.roll(np.roll(tempim2,-I1+agp2[0]/2, axis=0),-I2+agp2[1]/2,axis=1)

            xn[n] = I1
            yn[n] = I2

        a = imav2[2,6]
        a1 = im4[5,8]
        temp = np.copy(imav2)
        imav2 = (1-mix)*np.mean(im4,axis=2)+mix*temp

        b = imav2[2,6]

        result = agp[2]-np.sum(xn==xn2) # print convergence criterium
      #  print "convergence criterium =", result

        for n in range(agp[2]):
            im2[:,:,n] = np.roll(np.roll(im2[:,:,n],-I1+agp2[0]/2, axis=0),-I2+agp2[1]/2,axis=1)
            im3[:,:,n] = np.roll(np.roll(im3[:,:,n],-int(xn[n])+agp2[0]/2, axis=0),-int(yn[n])+agp2[1]/2,axis=1)
            Nf[:,:,n] = np.roll(np.roll(Nf[:,:,n],-int(xn[n])+agp2[0]/2, axis=0),-int(yn[n])+agp2[1]/2,axis=1)

    fig1 = plt.figure()
    ax00 = fig1.add_subplot(111)
    ax00.plot(xn,yn)
    save_data(xn, yn, file_path, 'fig1')
    #plt.tight_layout()
    #fig = plt.gcf()
    #plt.show()
    # Normalgr???sse anzeigen
    ftim2 = np.fft.fftshift(np.fft.fft(np.fft.fft(im2,axis =0),axis =1), axes = (0,1))

    im = ftim2[(agp2[0]-agp[0])/2:(agp2[0]+agp[0])/2,(agp2[1]-agp[1])/2:(agp2[1]+agp[1])/2,:]

    im = (1/float(ovsam))*(1/float(ovsam))*np.fft.ifft2(np.fft.ifftshift(im, axes =(0,1)),axes =(0,1)).real

    # Mittelwert bestimmen
    # nicht relevant
    imav2 = np.divide(np.sum(im2,axis =2),np.sum(Nf,axis =2));

    imav2[np.isnan(imav2)]=0;
    imav2[np.isinf(imav2)]=0;

    IND = np.argmax(imav2.flatten())
    Ix = IND/(imav2.shape[1])
    Iy = IND%(imav2.shape[1])


    imav2 = np.roll(np.roll(imav2,agp2[0]/2-Ix,axis = 0),agp2[1]/2-Iy, axis=1);
    im2 = np.roll(np.roll(im3,agp2[0]/2-Ix,axis = 0),agp2[1]/2-Iy, axis=1);
    # bis hier
    b = im3[135,135,:]

    imav3 = np.divide(np.sum(im3,axis = 2),np.sum(Nf,axis =2));
    a1 = imav3[136,:]
    imav3[np.isnan(imav3)]=0;
    imav3[np.isinf(imav3)]=0;
    IND = np.argmax(imav3.flatten())
    Ix = IND/(imav3.shape[1])
    Iy = IND%(imav3.shape[1])

    imav3 = np.roll(np.roll(imav3,agp2[0]/2-Ix,axis = 0),agp2[1]/2-Iy, axis=1);
    im3 = np.roll(np.roll(im3,agp2[0]/2-Ix,axis = 0),agp2[1]/2-Iy, axis=1);
    Nf = np.roll(np.roll(Nf,agp2[0]/2-Ix,axis = 0),agp2[1]/2-Iy, axis=1);

    #
    fig2 = plt.figure()
    ax00 = fig2.add_subplot(111)
    ax00.pcolor(imav3)
    ax00.set_title('averaged image')
    save_tiff(imav3, file_path, 'averaged_image')
    #plt.tight_layout()
    #fig = plt.gcf()
    #plt.show()

    ## psf (Mittelwert ???ber alle Winkel)
    temp = np.sort(R.flatten())
    r = [temp[0]]
    psfr = []
    r2 = []
    m = 0
    for n in xrange(1,temp.shape[0]):
        if (temp[n] != r[m]):
            m = m+1
            r.append(temp[n])

    nmax = np.sum(np.array(r) < rmax*ovsam) ;
    m = 0;
    for n in range(nmax):
        t1 = R==r[n]
        t2 = imav3[np.where(t1)]
        t3 = t2!=0
        t4 = np.sum(t3, axis = 0)
        t5 = np.sum(t4, axis = 0)

        if t5 != 0:
            tpsfr = np.mean(t2, axis = 0)
            if(tpsfr <0):
                tpsfr = 0

            psfr.append(tpsfr) 
            r2.append(r[n])
            m = m+1

    f = PchipInterpolator(r2,psfr)
    tend = X.shape[1]
    psf = f(np.absolute(X[0,tend/2-rmax*ovsam:tend/2+rmax*ovsam]))
    psfr = f(np.absolute(X3[0,:])) 

    psf = np.pad(psf,(agp2[0]/2-rmax*ovsam,agp2[0]/2-rmax*ovsam),'constant', constant_values =(0,0)); 
    tempsum = np.sum(np.multiply(psf,np.absolute(x2))* math.pi);
    psf = psf/tempsum * ovsam

    # 2D erstellen
    f2 = interp1d(x2,psf, kind = 'nearest', bounds_error=False, fill_value=0.)
    tf2 = f2(R2.flatten())
    PSF2 = np.reshape(tf2,R2.shape,order="F")
    temp = np.sum(PSF2.flatten())
    PSF2 = PSF2/temp *ovsam*ovsam

    # MTF
    MTF2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(PSF2))).real/(ovsam*ovsam)
    temp = MTF2.shape
    MTF = MTF2[temp[0]/2-ovsam*kadim[1]/2:temp[0]/2+ovsam*kadim[1]/2,temp[1]/2-ovsam*kadim[0]/2:temp[1]/2+ovsam*kadim[1]/2]


    fig3 = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    ax21 = fig3.add_subplot(gs[0,0])
    save_data(x2, psf, file_path, 'psf')

    ax21.plot(x2,psf)
    ax21.set_title('PSF')
    if(len(covr) !=0):
        y = np.max(psf)/np.max(covr)*np.array(covr)
        y = np.transpose(y)
        x = np.arange(-math.ceil(covr.shape[1]/2),covr.shape[1]-math.ceil(covr.shape[1]/2))
        ax21.plot(x,y)

    ax22 = fig3.add_subplot(gs[0,1])
    ax22_x = np.arange(-ovsam/2,ovsam/2,1/float(kadim[0]))
    ax22_y = MTF2[MTF2.shape[0]/2,:]
    ax22.plot(ax22_x, ax22_y)
    save_data(ax22_x, ax22_y, file_path, 'mtf')
    ax22.set_title('MTF')
    #
    #plt.tight_layout()
    #fig = plt.gcf()
    #plt.show()



    #
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    file_path = sys.argv[1]
    run_me(file_path)
