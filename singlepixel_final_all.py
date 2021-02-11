import numpy as np
from scipy.ndimage import imread
import ReadDMFile as dm3
from libtiff import TIFF
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import pickle
import os
import sys


def save_tiff(image, file_path, label):
    dir = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    base = os.path.splitext(base)[0]
    filename = os.path.join(dir, base + '_' + str(label) + '.tif')
    tif = TIFF.open(filename, mode='w')

    tif.write_image(image)


def save_data(x, y, file_path, label):
    dir = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    base = os.path.splitext(base)[0]
    filename = os.path.join(dir, base + '_' + str(label) + '.txt')
    fout = open(filename, 'w')
    fout_text = ''
    for i in range(len(y)):
        fout_text += str(x[i]) + ' ' + str(y[i]) + '\n'
    fout.write(fout_text)
    fout.close()


def save_data3(x, y, z, file_path, label):
    dir = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    base = os.path.splitext(base)[0]
    filename = os.path.join(dir, base + '_' + str(label) + '.txt')
    fout = open(filename, 'w')
    fout_text = ''
    for i in range(len(z)):
        fout_text += str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i] + '\n')
    fout.write(fout_text)
    fout.close()


def conv2(a, b):
    # fftconvolve returns the same value as in matlab in full mode, but for 'same' mode,
    # it selects center differently. This method deals with the difference.
    c = fftconvolve(a, b, 'full')
    n = c.shape
    m = a.shape
    dx = n[0] - m[0]
    dy = n[1] - m[1]

    endx = n[0] - math.floor(dx / 2)
    startx = endx - m[0]

    endy = n[1] - math.floor(dy / 2)
    starty = endy - m[1]
    return c[startx:endx, starty:endy]


def conv4(a, b):
    # fftconvolve returns the same value as in matlab in full mode, but for 'same' mode, it selects center differently. This method deals with the difference.
    c = fftconvolve(a, b, 'full')
    n = c.shape
    m = a.shape
    dx = n[0] - m[0]
    dy = n[1] - m[1]
    dz = n[2] - m[2]
    dt = n[3] - m[3]

    endx = n[0] - math.floor(dx / 2)
    startx = endx - m[0]

    endy = n[1] - math.floor(dy / 2)
    starty = endy - m[1]

    endz = n[2] - math.floor(dz / 2)
    startz = endz - m[2]

    endt = n[3] - math.floor(dt / 2)
    startt = endt - m[3]
    return c[startx:endx, starty:endy, startz:endz, startt:endt]


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
    return math.sqrt(variance)


def run_me(file_path):
    print('Starting Processing the stacked images')

    # Parameter
    ovsam = 4  # oversampling rate
    rmax = 7  # search radius for covarianze (<half image size)
    mix = 0.9  # mixing factor for subpixel localization
    convkrit = 5  # criterion for localization of subpixelposition
    covr = []
    ded_flag = 0  # low intensity flag (single electron counting mode)

    # Slowscan
    kadim = [64, 64]
    g = 18.5
    pixel = 14
    # NSFr = loadmat("C:/Users/lucaz/Documents/Bachelor_Arbeit/MATLAB/NSFr.mat")
    # covr = NSFr['covr']

    # Gitter bereitstellen
    print('Creating grid')
    temp = np.arange(-kadim[1] / 2, (kadim[1] * ovsam / 2) / ovsam, 1 / float(ovsam))

    [X2, Y2] = np.meshgrid(temp, temp)

    [temp, R2] = cart2pol(X2, Y2)  # polar grid for radially symmetric MTF
    temp1 = np.arange(-rmax, rmax + 1, 1)
    [X3, Y3] = np.meshgrid(temp1, temp1)
    temp2 = np.arange(-rmax * ovsam - 1, rmax * ovsam + 1, 1)
    [X4, Y4] = np.meshgrid(temp2, temp2)

    # Bild laden

    # file_path, _ = QtGui.QFileDialog.getOpenFileName(None, 'Please choose an image file','.')

    # file_path = "C:/Users/Luca Zangari/Documents/Uni/USB backup/TEM Messung 1 05.02.2016/DM3/TEST/test.dm3"
    # file_path = "C:/Users/Gapp/Desktop/Bachelor_Arbeit/Tiff Stacker and Processer/test out/output_stack.dm3"

    dx = 1
    dy = 1
    dz = 1
    if '.em' in file_path:
        expim = imread(file_path)
        tempx = np.array(expim)
        expim = tempx.astype(np.double)
    elif '.dm3' in file_path:
        dm3f = dm3.DM3(file_path)
        tempx = np.array(dm3f.imagedata)
        expim = tempx.astype(np.double)
        dx = int(dm3f.imagexdim[0])
        dy = int(dm3f.imageydim[0])
        dz = int(dm3f.imagezdim[0])
    elif '.tif' in file_path:
        tif = TIFF.open(file_path, 'r')
        dz = 0
        tempx = np.array([])
        for image in tif.iter_images():
            tempx = np.concatenate((tempx, np.ravel(image)))
            dz += 1
        dx = dy = np.sqrt(len(tempx) / dz)
        expim = tempx.astype(np.double)

    # Einzelelektronen herauslesen
    expim = np.reshape(expim, (dx, dy, dz), order="F")

    if ded_flag == 1:  # implement later
        agp = [32, 32]
        maxVal = 100
        nn = 0
        temp = []
        temp2 = np.zeros(shape=expim.shape)
        temp2[agp[0] / 2:agp[0] - agp[0] / 2, agp[1] / 2:agp[1] - agp[1] / 2] = expim[agp[0] / 2:agp[0] - agp[0] / 2,
                                                                                agp[1] / 2:agp[1] - agp[1] / 2]

        while maxVal > 20 and nn <= 600:
            print(nn)
            maxVal = np.max(temp2.flatten())
            I = np.argmax(temp2.flatten())
            # [row, col] = ind2sub(kadim, I);
            row = I / temp2.shape[0]
            col = I % temp2.shape[0]

            temp[:, :, nn] = temp2[row - agp[0] / 2 - 1:row + agp[0] / 2 - 1, col - agp[1] / 2 - 1:col + agp[1] / 2 - 1]

            temp2[row - agp[0] / 2 - 1:row + agp[0] / 2 - 1, col - agp[1] / 2 - 1:col + agp[1] / 2 - 1] = 0
            nn = nn + 1;

        expim = np.copy(temp)
        agp = expim.shape

    agp = expim.shape
    # *** SMP Modification: tuple multiplication causes replication not elementwise
    # multiplication.
    agp2 = tuple(np.array(agp) * ovsam)

    # Dunkelstrom entfernen
    dark = np.copy(expim)
    dark[1:agp[0] - 1, 1:agp[1] - 1, :] = 0

    temp = np.sum(dark, axis=0)
    temp1 = np.sum(temp, axis=0)

    dark = np.squeeze(temp1 / (2 * (agp[0] + agp[1] - 2)))

    for n in xrange(agp[2]):
        print('Clearing dark current')
        print(n)
        expim[:, :, n] = expim[:, :, n] - dark[n]

    # Belichtung normalisieren + Bilder entfernen (can be omitted)
    print('Integration...')
    temp = np.sum(expim, axis=0)
    temp1 = np.sum(temp, axis=0)
    print('intefrated image intensity')
    Iin = np.squeeze(temp1)  # integrated image intensity

    avIin = Iin.mean(axis=0)  # global average over whole series
    print('Global average over whole series')
    sIin = Iin.std(axis=0, ddof=1)  # global standard deviation over whole series
    temp = np.zeros((agp[0], agp[1], agp[2]))

    m = 0
    tresh = 0.4
    for n in xrange(agp[2]):
        if (abs(Iin[n] - avIin) <= tresh * avIin):
            temp[:, :, m] = expim[:, :, n]
            m = m + 1

    print ("m=" + str(m))
    expim = temp[:, :, 0:m]

    agp = expim.shape
    print agp

    Iin = np.sum(np.sum(expim, axis=0), axis=0)

    avIin = np.squeeze(Iin).mean(axis=0)

    sIin = np.squeeze(Iin).std(axis=0)

    if ded_flag == 1:
        g = avIin

    # gleitende Mittelwerte gleichsetzen
    im = np.copy(expim);

    # Gitter anlegen
    agp = np.array(expim.shape)
    agp2 = agp * ovsam

    x = np.arange(-agp[0] / 2, agp[1] / 2, 1)
    x2 = np.arange(-agp[0] / 2, agp[1] / 2, 1 / float(ovsam))
    [X, Y] = np.meshgrid(x2, x2)
    [temp, R] = cart2pol(X, Y)

    # vorl.Statistik bestimmen, (normieren)

    imav = im.mean(axis=2)

    imstd = im.std(axis=2)

    immax = np.max(np.max(im, axis=0), axis=0)
    #
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_y = np.squeeze(Iin)
    ax1.plot(ax1_y)
    save_data(xrange(len(ax1_y)), ax1_y, file_path, 'total_illumination')

    ax1.set_title('Total Illumination')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2_y = np.squeeze(immax)
    ax2.plot(ax2_y)
    save_data(xrange(len(ax2_y)), ax2_y, file_path, 'image_maximal_intensity')
    ax2.set_title('Image maximal intensity')
    #
    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()

    # Einzelaufnahmen ausrichten
    # oversampling Fourierraum

    temp = np.fft.fft2(imav)
    temp1 = np.fft.fftshift(temp)
    pw1 = (ovsam - 1) * agp[0] / 2
    pw2 = (ovsam - 1) * agp[1] / 2

    temp2 = np.pad(temp1, ((pw1, pw2), (pw1, pw2)), mode='constant', constant_values=0)
    temp3 = np.fft.ifftshift(temp2)
    temp4 = np.fft.ifft2(temp3)
    imav2 = ovsam * ovsam * (temp4.real)

    temp = np.fft.fft2(im, axes=(0, 1))
    temp1 = np.fft.fftshift(temp)
    temp2 = np.pad(temp1, ((pw1, pw2), (pw1, pw2), (0, 0)), mode='constant', constant_values=0)
    temp3 = np.fft.ifftshift(temp2)
    temp4 = np.fft.ifft2(temp3, axes=(0, 1))

    im2 = ovsam * ovsam * (temp4.real)

    im4 = np.copy(im2)

    # oversampling Ortsraum
    print('Oversampling')
    tempim = im.shape

    im3 = np.zeros(shape=(tempim[0], tempim[1], tempim[2], ovsam, ovsam))

    im3[:, :, :, math.floor(ovsam / 2), math.floor(ovsam / 2)] = np.copy(im)

    im3 = np.reshape(np.transpose(im3, axes=[3, 0, 4, 1, 2]), [ovsam * tempim[0], ovsam * tempim[1], tempim[2]],
                     order="F")

    temp = np.ones(shape=(tempim[0], tempim[1], tempim[2]))
    Nf = np.zeros(shape=(tempim[0], tempim[1], tempim[2], ovsam, ovsam))  # counts how many samples located on subpixel
    Nf[:, :, :, math.floor(ovsam / 2), math.floor(ovsam / 2)] = np.copy(temp)
    Nf = np.reshape(np.transpose(Nf, axes=[3, 0, 4, 1, 2]), [ovsam * tempim[0], ovsam * tempim[1], tempim[2]],
                    order="F")

    xn = np.ones(agp[2])
    yn = np.copy(xn)
    m = 0
    xn2 = xn + 1
    yn2 = yn + 1

    while (agp[2] - np.sum(xn == xn2) > convkrit) or (agp[2] - np.sum(yn == yn2) > convkrit):
        m = m + 1
        print('Kovarianz')
        xn2 = np.copy(xn)
        yn2 = np.copy(yn)

        maxin = np.argmax(imav2.flatten(1))

        Iy = maxin / (imav2.shape[1])
        Ix = maxin % (imav2.shape[1])

        imav2 = np.roll((np.roll(imav2, agp[0] / 2 - Ix, axis=0)), agp[1] / 2 - Iy, axis=1)
        print('Displacement')

        for n in xrange(agp[2]):
            print('Processing  {} ...'.format(n))
            # Fitmethode
            tempim2 = im2[:, :, n]
            psfcov = conv2(tempim2, imav2)
            IND = np.argmax(psfcov.flatten(1))

            I2 = IND / tempim2.shape[1]
            I1 = IND % tempim2.shape[1]

            # Verschieben
            im4[:, :, n] = np.roll(np.roll(tempim2, -I1 + agp2[0] / 2, axis=0), -I2 + agp2[1] / 2, axis=1)

            xn[n] = I1
            yn[n] = I2

        a = imav2[2, 6]
        a1 = im4[5, 8]
        temp = np.copy(imav2)
        imav2 = (1 - mix) * np.mean(im4, axis=2) + mix * temp

        b = imav2[2, 6]

        result = agp[2] - np.sum(xn == xn2)  # print convergence criterium
        #  print "convergence criterium =", result

        for n in range(agp[2]):
            # im2[:, :, n] = np.roll(np.roll(im2[:, :, n], -I1 + agp2[0] / 2, axis=0), -I2 + agp2[1] / 2, axis=1)
            im3[:, :, n] = np.roll(np.roll(im3[:, :, n], -int(xn[n]) + agp2[0] / 2, axis=0), -int(yn[n]) + agp2[1] / 2,
                                   axis=1)
            Nf[:, :, n] = np.roll(np.roll(Nf[:, :, n], -int(xn[n]) + agp2[0] / 2, axis=0), -int(yn[n]) + agp2[1] / 2,
                                  axis=1)

    fig1 = plt.figure()
    ax00 = fig1.add_subplot(111)
    ax00.plot(xn, yn)
    save_data(xn, yn, file_path, 'fig1')
    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()
    # Normalgr???sse anzeigen
    ftim2 = np.fft.fftshift(np.fft.fft(np.fft.fft(im2, axis=0), axis=1), axes=(0, 1))

    im = ftim2[(agp2[0] - agp[0]) / 2:(agp2[0] + agp[0]) / 2, (agp2[1] - agp[1]) / 2:(agp2[1] + agp[1]) / 2, :]

    im = (1 / float(ovsam)) * (1 / float(ovsam)) * np.fft.ifft2(np.fft.ifftshift(im, axes=(0, 1)), axes=(0, 1)).real

    # Mittelwert bestimmen
    # nicht relevant
    imav2 = np.divide(np.sum(im2, axis=2), np.sum(Nf, axis=2));

    imav2[np.isnan(imav2)] = 0;
    imav2[np.isinf(imav2)] = 0;

    IND = np.argmax(imav2.flatten())
    Ix = IND / (imav2.shape[1])
    Iy = IND % (imav2.shape[1])

    imav2 = np.roll(np.roll(imav2, agp2[0] / 2 - Ix, axis=0), agp2[1] / 2 - Iy, axis=1);
    im2 = np.roll(np.roll(im3, agp2[0] / 2 - Ix, axis=0), agp2[1] / 2 - Iy, axis=1);
    # bis hier
    b = im3[135, 135, :]

    imav3 = np.divide(np.sum(im3, axis=2), np.sum(Nf, axis=2));
    a1 = imav3[136, :]
    imav3[np.isnan(imav3)] = 0;
    imav3[np.isinf(imav3)] = 0;
    IND = np.argmax(imav3.flatten())
    Ix = IND / (imav3.shape[1])
    Iy = IND % (imav3.shape[1])

    imav3 = np.roll(np.roll(imav3, agp2[0] / 2 - Ix, axis=0), agp2[1] / 2 - Iy, axis=1);
    im3 = np.roll(np.roll(im3, agp2[0] / 2 - Ix, axis=0), agp2[1] / 2 - Iy, axis=1);
    Nf = np.roll(np.roll(Nf, agp2[0] / 2 - Ix, axis=0), agp2[1] / 2 - Iy, axis=1);

    #
    fig2 = plt.figure()
    ax00 = fig2.add_subplot(111)
    ax00.pcolor(imav3)
    print('Saving data from averaged image')
    ax00.set_title('averaged image')
    save_tiff(imav3, file_path, 'averaged_image')

    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()

    ## psf (Mittelwert ???ber alle Winkel)
    temp = np.sort(R.flatten())
    r = [temp[0]]
    psfr = []
    r2 = []
    m = 0
    for n in xrange(1, temp.shape[0]):
        if (temp[n] != r[m]):
            m = m + 1
            r.append(temp[n])

    nmax = np.sum(np.array(r) < rmax * ovsam);
    m = 0;
    for n in range(nmax):
        t1 = R == r[n]
        t2 = imav3[np.where(t1)]
        t3 = t2 != 0
        t4 = np.sum(t3, axis=0)
        t5 = np.sum(t4, axis=0)

        if t5 != 0:
            tpsfr = np.mean(t2, axis=0)
            if (tpsfr < 0):
                tpsfr = 0

            psfr.append(tpsfr)
            r2.append(r[n])
            m = m + 1

    f = PchipInterpolator(r2, psfr)
    tend = X.shape[1]
    psf = f(np.absolute(X[0, tend / 2 - rmax * ovsam:tend / 2 + rmax * ovsam]))
    psfr = f(np.absolute(X3[0, :]))

    psf = np.pad(psf, (agp2[0] / 2 - rmax * ovsam, agp2[0] / 2 - rmax * ovsam), 'constant', constant_values=(0, 0));
    tempsum = np.sum(np.multiply(psf, np.absolute(x2)) * math.pi);
    psf = psf / tempsum * ovsam

    # 2D erstellen
    f2 = interp1d(x2, psf, kind='nearest', bounds_error=False, fill_value=0.)
    tf2 = f2(R2.flatten())
    PSF2 = np.reshape(tf2, R2.shape, order="F")
    temp = np.sum(PSF2.flatten())
    PSF2 = PSF2 / temp * ovsam * ovsam

    # MTF
    MTF2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(PSF2))).real / (ovsam * ovsam)
    temp = MTF2.shape
    MTF = MTF2[temp[0] / 2 - ovsam * kadim[1] / 2:temp[0] / 2 + ovsam * kadim[1] / 2,
          temp[1] / 2 - ovsam * kadim[0] / 2:temp[1] / 2 + ovsam * kadim[1] / 2]

    fig3 = plt.figure()
    gs = gridspec.GridSpec(1, 2)
    ax21 = fig3.add_subplot(gs[0, 0])
    print('Saving PSF data')
    save_data(x2, psf, file_path, 'psf')
    ax21.plot(x2, psf)
    ax21.set_title('PSF')

    if (len(covr) != 0):
        y = np.max(psf) / np.max(covr) * np.array(covr)
        y = np.transpose(y)
        x = np.arange(-math.ceil(covr.shape[1] / 2), covr.shape[1] - math.ceil(covr.shape[1] / 2))
        ax21.plot(x, y)

    ax22 = fig3.add_subplot(gs[0, 1])
    ax22_x = np.arange(-ovsam / 2, ovsam / 2, 1 / float(kadim[0]))
    ax22_y = MTF2[MTF2.shape[0] / 2, :]
    ax22.plot(ax22_x, ax22_y)
    print('Saving MTF data')
    save_data(ax22_x, ax22_y, file_path, 'mtf')
    ax22.set_title('MTF')

    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()
    # # Einzelpixelkovarianz / integrierte Kovarianz bestimmen (oversampled)
    # Einzelpixelkovarianz / integrierte Kovarianz bestimmen (oversampled)
    # Korrelationsbereich ausschneiden
    print('Corrleation area')
    temp = np.transpose(im3, axes=[2, 0, 1])

    ars1 = temp.shape[1] / 2 - ovsam * rmax - math.floor(ovsam / 2)
    are1 = temp.shape[1] / 2 + ovsam * rmax + math.ceil(ovsam / 2)
    ars2 = temp.shape[2] / 2 - ovsam * rmax - math.floor(ovsam / 2)
    are2 = temp.shape[2] / 2 + ovsam * rmax + math.ceil(ovsam / 2)

    temp = temp[:, ars1:are1, ars2:are2]
    tdim = math.pow(ovsam * (2 * rmax + 1), 2)

    temp = np.reshape(temp, (agp[2], tdim), order="F")

    spcov = np.zeros(shape=(tdim, tdim))

    # Einzelpixelkovarianz
    print('Einzelpixelkovarianz')
    for n in xrange(ovsam * ovsam):
        x = n / ovsam
        y = n % ovsam

        if np.sum(Nf[x, y, :]) > 0:
            w = np.sum(Nf[x, y, :]) / agp[2] * ovsam * ovsam
            t1 = Nf[x, y, :] > 0
            t2 = np.squeeze(t1)
            t3 = temp[t2, :]
            if t3.shape[0] > 1:
                t4 = np.cov(np.transpose(t3))
            else:
                t4 = np.cov(t3)
            spcov = spcov + t4
    # kleine (negative) Werte filtern

    # Varianz + Rauschen der Varianz
    print('Varianz und Rauschen der Varianz')
    tdiag = np.diag(spcov)
    tdim = ovsam * (2 * rmax + 1)
    var = np.reshape(tdiag, (tdim, tdim), order="F")

    # Einzelpixelcorrelation
    print('Einzelpixelkorrelation')
    tvar = var.flatten()
    tcor = np.sqrt(np.absolute(tvar)) * np.sqrt(np.absolute(np.transpose(tvar)))

    spcor = np.divide(spcov, tcor)

    spcor[np.isinf(spcor)] = 0
    spcor[np.isnan(spcor)] = 0
    #
    #
    # Integrierte Kovarianz + NPS
    print('Integrierte Kovarianz + NPS')
    icov = np.zeros(shape=(2 * rmax + 1, 2 * rmax + 1))
    for nx in xrange(-rmax, rmax + 1):
        for ny in xrange(-rmax, rmax + 1):
            tdiag = np.diag(spcov, ovsam * nx * (ovsam * (2 * rmax + 1)) + ny * ovsam)
            icov[nx + rmax, ny + rmax] = sum(tdiag) / (ovsam * ovsam)

    NPS = np.absolute(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(icov))));

    # var + icov + nps (Achtung Fourierraum) ???ber Winkel mitteln + normalisieren +Fehler
    r = []

    [temp, R] = cart2pol(X4, Y4);
    temp = np.sort(R.flatten());
    r.append(temp[0]);
    m = 0
    for n in xrange(1, len(temp)):
        if (temp[n] != r[m]):
            m = m + 1
            r.append(temp[n])

    nmax = np.sum(np.array(r) < rmax * ovsam);
    m = 0;
    r2 = [];
    varr = []
    sigmavarr = []

    for n in xrange(nmax):
        teq = (R.transpose().flatten() == r[n])
        tvar = var.transpose().flatten()[teq]
        tvar1 = tvar.mean(axis=0)
        varr.append(tvar1)
        sigmavarr.append(tvar.std(axis=0))

        r2.append(r[n])
        m = m + 1

    # fig2 = plt.figure()
    # ax21 = fig2.add_subplot(111)
    # ax21.errorbar(r2[0:40],varr[0:40],sigmavarr[0:40])
    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()


    fvarr = PchipInterpolator(np.array(r2) / ovsam,
                              varr)  # differrent value at egdes incomparision with Matlab (https://github.com/scipy/scipy/issues/3453)
    varr = fvarr(np.absolute(X3[0, :]))

    DQEr = np.divide(np.multiply(psfr, psfr), varr)

    # icov
    print('icov')
    r = []
    [temp, R] = cart2pol(X3, Y3)
    temp = np.sort(R.flatten())
    r.append(temp[0])
    m = 0
    for n in xrange(1, len(temp)):
        if temp[n] != r[m]:
            m = m + 1
            r.append(temp[n])

    nmax = np.sum(np.array(r) < rmax)
    m = 0
    r2 = []
    icovr = []

    for n in xrange(nmax):
        t1 = R == r[n]
        t2 = icov[np.where(t1)]
        t3 = t2 != 0
        t4 = np.sum(t3, axis=0)
        t5 = np.sum(t4, axis=0)

        if t5 != 0:
            icovr.append(np.mean(icov[R == r[n]]))
            r2.append(r[n])
            m = m + 1

    ficovr = PchipInterpolator(r2, icovr)
    icovr = ficovr(np.absolute(X3[0, :]))

    # fig2 = plt.figure()
    # ax21 = fig2.add_subplot(111)
    # ax21.plot(X3[0,:],icovr/np.mean(np.squeeze(Iin)),'b')
    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()

    m = 0
    r2 = []
    NPSr = []
    for n in xrange(nmax):
        t1 = R == r[n]
        t2 = NPS[np.where(t1)]
        t3 = t2 != 0
        t4 = np.sum(t3, axis=0)
        t5 = np.sum(t4, axis=0)

        if t5 != 0:
            temnps = np.sum(np.sum(NPS[R == r[n]])) / np.sum(np.sum(NPS[R == r[n]] != 0))
            NPSr.append(temnps)
            r2.append(r[n])
            m = m + 1

    fnps = PchipInterpolator(r2, NPSr)
    NPSr = fnps(np.absolute(X3[0, :]))

    # auf kartesisches (ganzzahliges) Grid transformieren
    print('auf kartesisches (ganzzahliges) Grid transformieren')
    ndim = ovsam * (2 * rmax + 1)
    spcov = np.reshape(spcov, (ndim, ndim, ndim, ndim), order="F")
    spcor = np.reshape(spcor, (ndim, ndim, ndim, ndim), order="F")

    kerx = np.zeros(shape=(2, 2, 1, 1))
    kerx[0, 0, 0, 0] = 0.5
    kerx[1, 1, 0, 0] = 0.5
    kery = np.transpose(kerx, [2, 0, 3, 1])
    kerx = np.transpose(kerx, [0, 2, 1, 3])

    spcov = conv4(conv4(spcov, kerx), kery)
    spcor = conv4(conv4(spcor, kerx), kery)

    spcov = spcov[0:spcov.shape[0]:ovsam, 0:spcov.shape[1]:ovsam, 0:spcov.shape[2]:ovsam, 0:spcov.shape[3]:ovsam]
    spcor = spcor[0:spcor.shape[0]:ovsam, 0:spcor.shape[1]:ovsam, 0:spcor.shape[2]:ovsam, 0:spcor.shape[3]:ovsam]

    # Fehlerabschaetzung
    print('Fehlerabschaetzung')
    bound = 5
    temp = spcor[math.floor(spcor.shape[0] / 2) + bound - 1:spcor.shape[0], math.floor(spcor.shape[1] / 2),
           0:math.floor(spcor.shape[2] / 2) - bound, math.floor(spcor.shape[3] / 2)]

    corerr1 = np.std(temp.flatten())

    temp = spcor[math.floor(spcor.shape[0] / 2), math.floor(spcor.shape[1] / 2) + bound - 1:spcor.shape[1],
           math.floor(spcor.shape[2] / 2), math.floor(spcor.shape[3] / 2) - bound]
    corerr2 = np.std(temp.flatten())

    # principal component analyse
    temprscov = np.reshape(spcov, (math.pow(2 * rmax + 1, 2), math.pow(2 * rmax + 1, 2)), order="F")

    D, U = np.linalg.eigh(
        temprscov)  # Differrent value in comparision to matlab, but the values are multiple of each other(http://stackoverflow.com/questions/11691981/matlab-vs-python-eiga-b-vs-sc-linalg-eiga-b)

    IND = np.argsort(D)[::-1]

    U = U[:, IND]
    D = np.sort(D)[::-1]

    tempu = np.reshape(U, (2 * rmax + 1, 2 * rmax + 1, math.pow(2 * rmax + 1, 2)), order="F")

    U = np.absolute(tempu).real

    # fig2 = plt.figure()
    # plt.title('PCA');
    #
    # gs = gridspec.GridSpec(3, 3)
    # ax00 = fig2.add_subplot(gs[0,0])
    # ax00.plot(np.arange(20),D[0:20])
    # ax00.set_xticks(ticks=[0,10,20])
    #
    #
    # [X,Y] = np.meshgrid(np.arange(U.shape[0]), np.arange(U.shape[1]))
    #
    # ax01 = fig2.add_subplot(gs[0,1], projection='3d')
    # ax01.plot_surface(X,Y,U[:,:,0])
    #
    # ax02 = fig2.add_subplot(gs[0,2], projection='3d')
    # ax02.plot_surface(X,Y,U[:,:,1])
    #
    # ax10 = fig2.add_subplot(gs[1,0], projection='3d')
    # ax10.plot_surface(X,Y,U[:,:,2])
    #
    # ax11 = fig2.add_subplot(gs[1,1], projection='3d')
    # ax11.plot_surface(X,Y,U[:,:,3])
    #
    # ax12 = fig2.add_subplot(gs[1,2], projection='3d')
    # ax12.plot_surface(X,Y,U[:,:,4])
    #
    # ax20 = fig2.add_subplot(gs[2,0], projection='3d')
    # ax20.plot_surface(X,Y,U[:,:,5])
    #
    # ax21 = fig2.add_subplot(gs[2,1], projection='3d')
    # ax21.plot_surface(X,Y,U[:,:,6])
    #
    # ax22 = fig2.add_subplot(gs[2,2], projection='3d')
    # ax22.plot_surface(X,Y,U[:,:,7])


    # SNR + DQE
    # Ortsraum
    print('Signal to Noise Ratio and DQE')
    print('Erstelle Ortsraum')
    s0 = imav3.shape[0] / 2 - ovsam * rmax - math.floor(ovsam / 2)
    e0 = imav3.shape[0] / 2 + ovsam * rmax + math.ceil(ovsam / 2)
    s1 = imav3.shape[1] / 2 - ovsam * rmax - math.floor(ovsam / 2)
    e1 = imav3.shape[1] / 2 + ovsam * rmax + math.ceil(ovsam / 2)
    timav3 = imav3[s0:e0, s1:e1]
    SNR = np.divide(np.multiply(timav3, timav3), var)

    SNR[np.isnan(SNR)] = 0
    SNR[np.isinf(SNR)] = 0

    SNRi = np.ones(shape=SNR.shape)
    SNRi[round(SNRi.shape[0] / 2), round(SNRi.shape[1] / 2 + 1)] = np.mean(np.squeeze(Iin)) / g

    DQE = np.divide(SNR, SNRi)
    print(DQE)
    # Fourierraum
    print('Erstelle Furrierraum')
    s0 = imav3.shape[0] / 2 - ovsam * rmax - math.floor(ovsam / 2)
    e0 = imav3.shape[0] / 2 + ovsam * rmax + math.ceil(ovsam / 2)
    s1 = imav3.shape[1] / 2 - ovsam * rmax - math.floor(ovsam / 2)
    e1 = imav3.shape[1] // 2 + ovsam * rmax + math.ceil(ovsam / 2)
    timav3 = imav3[s0:e0, s1:e1]
    t1 = np.absolute(np.fft.fft2(timav3))
    t2 = np.absolute(np.fft.fft2(var))
    t3 = np.divide(np.multiply(t1, t1), t2)

    SNRf = (1 / float(ovsam * ovsam)) * np.fft.fftshift(t3);
    SNRf[np.isnan(SNRf)] = 0
    SNRf[np.isinf(SNRf)] = 0
    SNRif = np.mean(np.squeeze(Iin)) / g
    DQEf = np.divide(SNRf, SNRif)

    temp = np.fft.ifftshift(np.fft.fft2(imav3) / agp[0] / agp[1])

    temp = temp[temp.shape[0] / 2 - rmax:temp.shape[0] / 2 + 1 + rmax,
           temp.shape[1] / 2 - rmax:temp.shape[1] / 2 + 1 + rmax]

    temp2 = NPS[NPS.shape[0] / 2 - rmax:NPS.shape[0] / 2 + rmax + 1,
            NPS.shape[1] / 2 - rmax:NPS.shape[1] / 2 + 1 + rmax]

    SNRfold = np.divide(np.multiply(np.absolute(temp), np.absolute(temp)), temp2)

    SNRfold[np.isnan(SNRfold)] = 0;
    SNRfold[np.isinf(SNRfold)] = 0;

    DQEfold = SNRfold / SNRif * np.mean(np.squeeze(Iin));

    fig1 = plt.figure()
    gs = gridspec.GridSpec(2, 2)

    [X, Y] = np.meshgrid(np.arange(var.shape[0]), np.arange(var.shape[1]))

    ax00 = fig1.add_subplot(gs[0, 0], projection='3d')
    ax00.plot_surface(X, Y, var)
    print('Saving Variance data')
    ax00.set_title('variance');

    [X, Y] = np.meshgrid(np.arange(SNR.shape[0]), np.arange(SNR.shape[1]))

    ax01 = fig1.add_subplot(gs[0, 1], projection='3d')
    ax01.plot_surface(X, Y, SNR)
    print('Saving SNR data')
    ax01.set_title('Signal2noise');

    [X, Y] = np.meshgrid(np.arange(icov.shape[0]), np.arange(icov.shape[1]))

    ax10 = fig1.add_subplot(gs[1, 0], projection='3d')
    ax10.plot_surface(X, Y, icov / np.mean(np.squeeze(Iin)))
    ax10.set_title('normalized integrated covariance');

    [X, Y] = np.meshgrid(np.arange(DQE.shape[0]), np.arange(DQE.shape[1]))

    ax11 = fig1.add_subplot(gs[1, 1], projection='3d')
    ax11.plot_surface(X, Y, DQE)
    ax11.set_title('DQE');

    # Kovarianz, Korrelation
    subman = np.squeeze(spcov[:, math.floor(spcov.shape[1] / 2), :, math.floor(spcov.shape[3] / 2)]);
    fig2 = plt.figure()

    [X, Y] = np.meshgrid(np.arange(subman.shape[0]), np.arange(subman.shape[1]))

    ax00 = fig2.add_subplot(gs[0, 0], projection='3d')
    ax00.plot_surface(X, Y, subman)
    ax00.set_title('x,x covariance at y,y=0')
    # axis square

    subman = np.squeeze(spcor[:, math.floor(spcor.shape[1] / 2), :, math.floor(spcor.shape[3] / 2)])

    [X, Y] = np.meshgrid(np.arange(subman.shape[0]), np.arange(subman.shape[1]))

    ax01 = fig2.add_subplot(gs[0, 1], projection='3d')
    ax01.plot_surface(X, Y, subman)
    ax01.set_title('x,x correlation at y,y=0')
    # axis square

    pickle.dump(subman, open("corxx4.p", "wb"))

    subman = np.squeeze(spcov[math.floor(spcov.shape[0] / 2), :, math.floor(spcov.shape[2] / 2), :])
    [X, Y] = np.meshgrid(np.arange(subman.shape[0]), np.arange(subman.shape[1]))

    ax10 = fig2.add_subplot(gs[1, 0], projection='3d')
    ax10.plot_surface(X, Y, subman)
    ax10.set_title('y,y covariance at x,x=0')
    # axis square

    subman = np.squeeze(spcor[math.floor(spcor.shape[0] / 2), :, math.floor(spcor.shape[2] / 2), :])
    [X, Y] = np.meshgrid(np.arange(subman.shape[0]), np.arange(subman.shape[1]))

    ax11 = fig2.add_subplot(gs[1, 1], projection='3d')
    ax11.plot_surface(X, Y, subman)
    ax11.set_title('y,y correlation at x,x=0');
    # axis square
    pickle.dump(subman, open("coryy4.p", "wb"))

    fig3 = plt.figure()
    ax00 = fig3.add_subplot(111)
    ax00.plot(X3[0, :], icovr / np.mean(np.squeeze(Iin)))
    ax00.set_title('normalized integrated covariance');

    fig4 = plt.figure()
    gs = gridspec.GridSpec(1, 2)

    [X, Y] = np.meshgrid(np.arange(DQEf.shape[0]), np.arange(DQEf.shape[1]))

    ax00 = fig4.add_subplot(gs[0, 0], projection='3d')
    ax00.plot_surface(X, Y, DQEf)
    ax00.set_title('DQEnew');
    # axis square

    [X, Y] = np.meshgrid(np.arange(DQEfold.shape[0]), np.arange(DQEfold.shape[1]))

    ax01 = fig4.add_subplot(gs[0, 1], projection='3d')
    ax01.plot_surface(X, Y, DQEfold)
    ax01.set_title('DQEold');

    fig5 = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    [X, Y] = np.meshgrid(np.arange(temp.shape[0]), np.arange(temp.shape[1]))
    ax00 = fig5.add_subplot(gs[0, 0], projection='3d')
    ax00.plot_surface(X, Y, np.absolute(temp))

    fig6 = plt.figure()
    ax00 = fig6.add_subplot(111)
    ax00.plot(X3[0, :], varr)

    fig7 = plt.figure()
    ax00 = fig7.add_subplot(111)
    ax00.plot(X3[0, :], DQEr)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_path = sys.argv[1]
    run_me(file_path)
