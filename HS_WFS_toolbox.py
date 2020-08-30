import os
import json
from struct import *
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt

def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError ('Input vectors y_axis and x_axis must have same length')

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)


    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]
    
def fast_gaussian_fit(y, x):
    """ Fast Gaussian filter.
    
    Keyword arguments:
    y -- density array.
    x -- x position array.
    
    Returns:
    A, mu, sigma, C -- fitted Gaussian parameters.
    """    
    w = len(y) # size of the array
    avg = np.mean(y) # average of the density
    M = np.max(y) # maximum density
    m = np.average(np.sort(y)[0:w//3]) # average mimimum density
    u = np.average(x, weights=y) # exception of x
    eta = 1 # erf factor
    
    A = M-m
    C = m
    _a = (avg-m)*w
    sigma = _a/(eta*np.sqrt(2*np.pi)*A)
    mu = u*avg*w/_a
    
    return A, mu, sigma, C

def peak_gaussian_fit(y, x):
    """ Peak Gaussian filter.
    
    Seek for the peak of each direction, use the peak position as
    the mass center position.
    
    Keyword arguments:
    y -- density array.
    x -- x position array.
    
    Returns:
    A, mu, sigma, C -- fitted parameters.
    """    
    w = len(y) # size of the array
    avg = np.mean(y) # average of the density
    M = np.max(y) # maximum density
    m = np.average(np.sort(y)[0:w//3]) # average mimimum density
    eta = 1 # erf factor
    
    A = M-m
    C = m
    _a = (avg-m)*w
    sigma = _a/(eta*np.sqrt(2*np.pi)*A)
    _y = y-C
    _y[(_y < 0)] = 0
    i_peak = np.argmax(_y)
    i_lower = np.max([i_peak-2, 0])
    i_upper = i_peak+3
    mu = np.average(x[i_lower:i_upper], weights=_y[i_lower:i_upper])
    
    return A, mu, sigma, C
    
def get_main_peak_index(p):
    p_diff = np.diff(p)
    n = len(p_diff)
    
    avg = np.mean(np.sort(p_diff)[:n//2])
    invalid = p_diff > 2*avg
    breakpoint = np.hstack((-1, np.arange(n)[invalid], n))
    lens = np.diff(breakpoint)
    i_main = np.argmax(lens)
    
    return np.arange(breakpoint[i_main]+1, breakpoint[i_main+1]+1)

def get_reference(p, n):
    i_p = get_main_peak_index(p)
    j_n = get_main_peak_index(n)
    len_p = len(i_p)
    len_n = len(j_n)
    len_t = len_p+len_n-1
    _p = np.hstack((p[i_p], np.nan*np.ones(len_n-1)))
    _n = np.hstack((np.nan*np.ones(len_p-1), n[j_n]))
    
    imin = []
    dmin = []
    for i in range(len_t):
        diff = np.abs(_p[:len_t-i]-_n[i:])
        _i = np.nanargmin(diff)
        imin.append(_i)
        dmin.append(diff[_i])
    offset = np.argmin(dmin)
    i = imin[offset]
    j = offset+i-len_p+1
    
    return i_p[0]+i, j_n[0]+j
        
def if_align(i, j, p, n, thres=8, shift=False):
    if shift:
        if i > 0 and j > 0:
            delta = p[i-1]-n[j-1]
        else:
            delta = 0
        align = (np.abs(p[i]-n[j]-delta) <= thres)
    else:
        align = (np.abs(p[i]-n[j]) <= thres)

    return align

def if_exists(i, j, p, n):
    flag = [0, 0]
    try:
        pi = p[i]
        flag[0] = 1
    except IndexError:
        pass
    try:
        nj = n[j]
        flag[1] = 1
    except IndexError:
        pass
    
    return 2*flag[1]+flag[0]

def is_breakpoint(i, p):
    p_diff = np.diff(p)    
    avg = np.mean(np.sort(p_diff)[:len(p_diff)//2])
    
    if i:
        return p_diff[i-1] > 2*avg
    else:
        return True
    
def align_peaks(ppos, pval, npos, nval, final=False):
    try:
        ppos.shape[1]
    except IndexError:
        ppos = ppos.reshape(1, -1)
        pval = pval.reshape(1, -1)
    # Get reference index
    i0, j0 = get_reference(ppos[-1], npos)
    
    # The plus part (->)
    _ppos = ppos[:, i0:]
    _pval = pval[:, i0:]
    _npos = npos[j0:]
    _nval = nval[j0:]
    pos_plus, val_plus = align_peaks_A(1, 1, _ppos, _pval, _npos, _nval, final)
    
    # The minus part (<-)
    p0 = ppos[-1, i0]
    _ppos = 2*p0-ppos[:, i0::-1]
    _pval = pval[:, i0::-1]
    _npos = 2*p0-npos[j0::-1]
    _nval = nval[j0::-1]
    pos_minus, val_minus = align_peaks_A(1, 1, _ppos, _pval, _npos, _nval, final)
    
    # Combine the two parts
    pos = np.hstack((2*p0-pos_minus[:, ::-1], pos_plus[:, 1:]))
    val = np.hstack((val_minus[:, ::-1], val_plus[:, 1:]))
    
    return pos, val

def align_peaks_A(i, j, ppos, pval, npos, nval, final=False):
    r = ppos.shape[0]
    
    flag = if_exists(i, j, ppos[-1], npos)
    if flag == 0:
        if final:
            return ppos, pval
        else:
            return np.vstack((ppos, npos)), np.vstack((pval, nval))
    elif flag == 1:
        if final:
            ppos = np.delete(ppos, i, axis=1)
            pval = np.delete(pval, i, axis=1)
            return align_peaks_A(i, j, ppos, pval, npos, nval, final)
        else:
            npos = np.insert(npos, j, ppos[-1, i])
            nval = np.insert(nval, j, np.nan)
            i += 1
            j += 1
            return align_peaks_A(i, j, ppos, pval, npos, nval, final)
    elif flag == 2:
        ppos = np.insert(ppos, i, npos[j]*np.ones(r), axis=1)
        pval = np.insert(pval, i, np.nan*np.ones(r), axis=1)
        i += 1
        j += 1
        return align_peaks_A(i, j, ppos, pval, npos, nval, final)
    else:
        if if_align(i, j, ppos[-1], npos, 8, final):
            i += 1
            j += 1
            return align_peaks_A(i, j, ppos, pval, npos, nval, final)
        else:
            return align_peaks_B(i, j, ppos, pval, npos, nval, final)
        
def align_peaks_B(i, j, ppos, pval, npos, nval, final=False):
    r = ppos.shape[0]
    
    if ppos[-1, i] < npos[j]:
        if if_exists(i+1, 0, ppos[-1], []):
            if if_align(i+1, j, ppos[-1], npos, 8, final):
                if is_breakpoint(j, npos):
                    if final:
                        ppos = np.delete(ppos, i, axis=1)
                        pval = np.delete(pval, i, axis=1)
                        i += 1
                        j += 1
                        return align_peaks_A(i, j, ppos, pval, npos, nval, final)
                    else:
                        npos = np.insert(npos, j, ppos[-1, i])
                        nval = np.insert(nval, j, np.nan)
                        i += 2
                        j += 2
                        return align_peaks_A(i, j, ppos, pval, npos, nval, final)
                else:
                    ppos = np.delete(ppos, i, axis=1)
                    pval = np.delete(pval, i, axis=1)
                    i += 1
                    j += 1
                    return align_peaks_A(i, j, ppos, pval, npos, nval, final)
            else:
                if is_breakpoint(j, npos):
                    if final:
                        ppos = np.delete(ppos, i, axis=1)
                        pval = np.delete(pval, i, axis=1)
                        return align_peaks_B(i, j, ppos, pval, npos, nval, final)
                    else:
                        npos = np.insert(npos, j, ppos[-1, i])
                        nval = np.insert(nval, j, np.nan)
                        i += 1
                        j += 1
                        return align_peaks_B(i, j, ppos, pval, npos, nval, final)
                else:
                    i += 1
                    j += 1
                    return align_peaks_A(i, j, ppos, pval, npos, nval, final)
        else:
            if is_breakpoint(j, npos):
                if final:
                    ppos = np.delete(ppos, i, axis=1)
                    pval = np.delete(pval, i, axis=1)
                    return align_peaks_A(i, j, ppos, pval, npos, nval, final)
                else:
                    npos = np.insert(npos, j, ppos[-1, i])
                    nval = np.insert(nval, j, np.nan)
                    i += 1
                    j += 1
                    return align_peaks_A(i, j, ppos, pval, npos, nval, final)
            else:
                i += 1
                j += 1
                return align_peaks_A(i, j, ppos, pval, npos, nval, final)
    else:
        if if_exists(j+1, 0, npos, []):
            if if_align(i, j+1, ppos[-1], npos, 8, final):
                if is_breakpoint(i, ppos[-1]):
                    ppos = np.insert(ppos, i, npos[j]*np.ones(r), axis=1)
                    pval = np.insert(pval, i, np.nan*np.ones(r), axis=1)
                    i += 2
                    j += 2
                    return align_peaks_A(i, j, ppos, pval, npos, nval, final)
                else:
                    if final:
                        ppos = np.insert(ppos, i, npos[j]*np.ones(r), axis=1)
                        pval = np.insert(pval, i, np.nan*np.ones(r), axis=1)
                        i += 2
                        j += 2
                        return align_peaks_A(i, j, ppos, pval, npos, nval, final)
                    else:
                        npos = np.delete(npos, j)
                        nval = np.delete(nval, j)
                        i += 1
                        j += 1
                        return align_peaks_A(i, j, ppos, pval, npos, nval, final)
            else:
                if is_breakpoint(i, ppos[-1]):
                    ppos = np.insert(ppos, i, npos[j]*np.ones(r), axis=1)
                    pval = np.insert(pval, i, np.nan*np.ones(r), axis=1)
                    i += 1
                    j += 1
                    return align_peaks_B(i, j, ppos, pval, npos, nval, final)
                else:
                    i += 1
                    j += 1
                    return align_peaks_A(i, j, ppos, pval, npos, nval, final)
        else:
            if is_breakpoint(i, ppos[-1]):
                ppos = np.insert(ppos, i, npos[j]*np.ones(r), axis=1)
                pval = np.insert(pval, i, np.nan*np.ones(r), axis=1)
                i += 1
                j += 1
                return align_peaks_A(i, j, ppos, pval, npos, nval, final)
            else:
                i += 1
                j += 1
                return align_peaks_A(i, j, ppos, pval, npos, nval, final)
                
def complete_peaks(peaks):
    """ Fill in the missing peaks.

    When the distance between two nearby peaks is too large, insert expected
    peaks between them.

    Keyword arguments:
    peaks -- array of peak positions and peak values.

    Returns:
    peaks -- the complete peak positions and peak values.
    """
    peak_pos = peaks[:, 0]
    pos_diff = np.diff(peak_pos)
    n = len(pos_diff)
    x = np.arange(n)
    y = pos_diff

    avg = np.mean(np.sort(y)[:n//2])
    invalid = (y >= 1.5*avg) & (y < 2.5*avg)
    breakpoint = x[invalid]
    gaps = y[invalid]
    inserts = np.round(gaps/avg-1).astype(int)

    _peak_pos = np.copy(peak_pos)
    _peak_value = np.copy(peaks[:, 1])
    for i in range(len(breakpoint))[::-1]:
        ib = breakpoint[i]
        _m_peak_pos = np.linspace(_peak_pos[ib], _peak_pos[ib+1], inserts[i]+2)[1:-1]
        _peak_pos = np.hstack((_peak_pos[:ib+1], _m_peak_pos, _peak_pos[ib+1:]))
        _peak_value = np.hstack((_peak_value[:ib+1], np.nan*np.ones(inserts[i]),
                                 _peak_value[ib+1:]))

    return np.hstack((_peak_pos.reshape(-1, 1), _peak_value.reshape(-1, 1)))

def clear_side_peaks(peaks):
    """ Clear the invalid side peaks.
    
    Keyword arguments:
    peaks -- array of peak positions and peak values.
    
    Returns:
    peaks -- the filtered peaks.
    """
    peak_pos = peaks[:, 0]
    pos_diff = np.diff(peak_pos)
    n = len(pos_diff)
    x = np.arange(n)
    y = pos_diff
    
    if n:
        avg = np.mean(np.sort(y)[:n//2])
        invalid = y < 0.6*avg
        
        offset = []
        for i in x[invalid]:
            y1 = y[i]
            try:
                y0 = y[i-1]
            except IndexError:
                y0 = 0
            try:
                y2 = y[i+1]
            except IndexError:
                y2 = 0
            offset.append(0 if (np.abs(y0+y1-avg) < np.abs(y1+y2-avg)) else 1)
            
        i_invalid = x[invalid]+np.array(offset)
        _peak_pos = np.copy(peak_pos)
        _peak_pos[i_invalid.astype(int)] = -1
        
        return peaks[_peak_pos != -1, :]
    else:
        return peaks

def clear_peak_groups(peaks, minlen=3):
    """ Clear the invalid peak groups.
    
    Keyword arguments:
    peaks -- array of peak positions and peak values.
    minlen -- minimum length of the valid peak group. [3]
        Ignore the peak group of which the length is shorter than minlen.
    
    Returns:
    peaks -- the filtered peaks.
    """
    peak_pos = peaks[:, 0]
    pos_diff = np.diff(peak_pos)
    n = len(pos_diff)
    x = np.arange(n)
    y = pos_diff
    
    if n:
        avg = np.mean(np.sort(y)[:n//2])
        invalid = y > 2*avg
        breakpoint = np.hstack((-1, x[invalid], n))
        lens = np.diff(breakpoint)
        i_valid = np.arange(len(lens))[lens >= minlen]
        
        if len(i_valid):
            index = np.hstack([np.arange(breakpoint[i]+1, breakpoint[i+1]+1)\
                               for i in i_valid])
            return peaks[index, :]
        else:
            # No valid peaks
            return np.array([[]])
    else:
        return peaks

def grid_uniformor(peaks, minlen=3, show=False):
    """ Uniform the grid.
    
    Ignore the invalid peak groups & peaks and fill in the missing peaks.
    
    Keyword arguments:
    peaks -- array of peak positions and peak values, shape N*2.
    minlen -- minimum length of the valid peak group. [3]
        Ignore the peak group of which the length is shorter than minlen.
    show -- if show the uniformed peaks. [False]
    
    Returns:
    peaks -- the uniformed peaks.
    """
    # Clear the invalid peak groups
    peaks = clear_peak_groups(peaks, minlen)
    # Clear the invalid peaks (that are too close to the standard peaks)
    peaks = clear_side_peaks(peaks)
    # Fill in the missing peaks
    peaks = complete_peaks(peaks)
    
    return peaks

def get_curve_grid(pic, peak_w, peak_h, direction='x', signlevel=0):
    n = peak_h.shape[1] if (direction == 'x') else peak_w.shape[1]
    
    for i in range(n):
        if i == 0:
            peak = HS_WFS_get_slice_peaks(0, pic, peak_w[0], peak_h[0], 
                                          signlevel=signlevel, lookahead=5, 
                                          direction=direction, show=0, uniform=1,
                                          minlen=5)
            peak_pos = peak[0]
            peak_val = peak[1]
        else:
            _peak = HS_WFS_get_slice_peaks(i, pic, peak_w[0], peak_h[0],
                                           signlevel=signlevel, lookahead=5, 
                                           direction=direction, show=0, uniform=1,
                                           minlen=5)
            peak_pos, peak_val = align_peaks(peak_pos, peak_val, _peak[0], _peak[1])
    
    # Get valid peaks index
    peak_g = peak_w if (direction == 'x') else peak_h
    peak_pos, peak_val = align_peaks(peak_pos, peak_val, peak_g[0], peak_g[1], final=1)
    
    peak_pos[np.isnan(peak_val)] = np.nan
    
    return peak_pos
    
def HS_WFS_cmap_reader(filename, path=''):
    """ Read the classic HS WFS colormap.
    
    Keyword arguments:
    filename -- the colormap file name, with extension (.json).
    path -- the path to the colormap file. ['']
    
    Returns:
    cmap -- the colormap of matplotlib format.
    """
    cmfile = os.path.join(path, filename)
    with open(cmfile) as f:
        cmdata = json.load(f)
        colors = (np.array(cmdata)/255)
        colors[:, -1] = 1
        cmap = mpl.colors.ListedColormap(colors, os.path.splitext(filename)[0])
        
    return cmap

def HS_WFS_data_reader(filename, path=''):
    """ Read the standard HS WFS image data.
    
    Keyword arguments:
    filename -- the data file name, with extension (.rawb or .imx or .iss).
    path -- the path to the data file. ['']
    
    Returns:
    pics -- the list of the image matrix.
    info -- the information of the data.
    """
    ext = os.path.splitext(filename)[1]
    if ext == '.rawb':
        pic = _HS_WFS_frame_reader(filename, path)
        height, width = pic.shape
        
        pics = [pic]
        info = {'frame num': 1, 'width': width, 'height': height}
    elif ext in ['.imx', '.iss']:
        pics, info = _HS_WFS_series_reader(filename, path)
    else:
        print('Sorry, data format {} is not supported!'.format(ext))
        pics, info = [], {}
    
    return pics, info    
    
def _HS_WFS_frame_reader(filename, path=''):
    """ Read single frame standard HS WFS image data.
    
    Data format: pixel: 1 byte. width, height: 4 byte (1 int).
    
    Keyword arguments:
    filename -- the data file name, with extension (.rawb).
    path -- the path of the data files. ['']
    
    Returns:
    pic -- the image matrix.
    """
    with open(os.path.join(path, filename), 'rb') as f:
        data = f.read()
    # Get image size info
    info = Struct('ii')
    infosize = info.size
    width, height = info.unpack(data[-infosize:])
    # Get the image data
    frame = Struct('{:d}B'.format(width*height))
    framesize = frame.size
    pic = np.array(frame.unpack(data[:framesize])).reshape((height, width))
    
    return pic

def _HS_WFS_series_reader(filename, path=''):
    """ Read a series of standard HS WFS image data.
    
    Keyword arguments:
    filename -- the image series data file name, with extension (.imx or .iss).
    path -- the path of the data files. ['']
    
    Returns:
    pics -- the list of the image matrix.
    info_list -- the information of the image series.
    """
    with open(os.path.join(path, filename), 'rb') as f:
        data = f.read()
    # Get image series info
    info = Struct('if2i2f2h4Bf')
    infosize = info.size
    info_list = info.unpack(data[-infosize:])
    
    frame_num = info_list[0]
    width = info_list[6]
    height = info_list[7]
    # Get the image series data
    ext = os.path.splitext(filename)[1]
    datatype = 'B' if ext == '.iss' else 'H' # '.iss': pixel format byte, '.imx': pixel format word
    frame = Struct('{0:d}{1}'.format(width*height, datatype))
    framesize = frame.size
    
    pics = []
    for i in range(frame_num):
        pics.append(np.array(frame.unpack(data[i*framesize:(i+1)*framesize])).reshape((height, width)))
    
    return pics, info_list

def HS_WFS_get_peaks(pic, lookahead=10, signlevel=0, uniform=True, minlen=3, 
                     show=False, save=False):
    """ Get peaks of the HS WFS image.
    
    Project the image to x and y directions, then detect the peaks in both directions.
    
    Keyword arguments:
    pic -- the image matrix.
    lookahead -- how many points to look ahead when detect the peaks. [10]
    signlevel -- filter out the trivial peaks that satisfy: [0]
        (peak-avg)/avg < signlevel
    uniform -- if apply the grid uniformor. [True]
    minlen -- minimum length of the valid peak group. [3]
        Ignore the peak group of which the length is shorter than minlen.
    show -- if show the detected peaks. [False]
    save -- if save the figure of detected peaks. [False]
    
    Returns:
    peak_max_w, peak_max_h -- peak positions and values of the x and y directions.
    """
    height, width = pic.shape
    
    if show:
        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        
        xlims = [width, height]
        for i, ax in enumerate(axs):
            ax.set(xlabel='pixel number (1)', ylabel='projected density (arb. units.)', 
                   xlim=(0, xlims[i]))

    # Width
    x = np.arange(width)
    y = np.mean(pic, 0)

    peak_max, peak_min = peakdetect(y, x, lookahead)
    peak_max = np.array(peak_max)
    # Filter out the trivial peaks
    def signfilter(i, r=20):
        if i == 0:
            x0 = x[0]
            x1, x2 = peak_max[0:2, 0]
        else:
            try:
                x0, x1, x2 = peak_max[i-1:i+2, 0]
            except ValueError:
                x0, x1 = peak_max[i-1:i+1, 0]
                x2 = x[-1]
        x0 = x1-np.min([r, x1-x0])
        x2 = x1+np.min([r, x2-x1])
        
        xlower, xupper = (x0+x1)/2, (x1+x2)/2
        index = (x <= xupper) & (x >= xlower)
        yn = y[index]
        avg = (yn[0]+yn[-1])/2
        peak = peak_max[i, 1]

        if (peak-avg)/avg >= signlevel:
            return True
        else:
            return False

    valid = np.array([signfilter(i) for i in range(peak_max.shape[0])])
    peak_max_w = peak_max[valid, :]

    if uniform:
        peak_max_w = grid_uniformor(peak_max_w, minlen=minlen)
    
    peak_pos_w = peak_max_w[:, :1]
    peak_value_w = peak_max_w[:, 1:]
    
    if show:
        axs[0].plot(x, y, 'b-', label='horizontal\nlookahead = {0}\nsignlevel = {1}'.format(lookahead, signlevel))
        if len(peak_pos_w):
            axs[0].plot(peak_pos_w, peak_value_w, 'rx')

    # Height
    x = np.arange(height)
    y = np.mean(pic, 1)

    peak_max, peak_min = peakdetect(y, x, lookahead)
    peak_max = np.array(peak_max)
    
    valid = np.array([signfilter(i) for i in range(peak_max.shape[0])])
    peak_max_h = peak_max[valid, :]
    
    if uniform:
        peak_max_h = grid_uniformor(peak_max_h, minlen=minlen)
        
    peak_pos_h = peak_max_h[:, :1]
    peak_value_h = peak_max_h[:, 1:]
    
    if show:
        axs[1].plot(x, y, 'g-', label='vertical\nlookahead = {0}\nsignlevel = {1}'.format(lookahead, signlevel))
        if len(peak_pos_h):
            axs[1].plot(peak_pos_h, peak_value_h, 'rx')
        
        for ax in axs:
            ax.legend(loc=0)
        fig.tight_layout()
        if save:
            fig.savefig('peaks.pdf', bbox_inches='tight')
        plt.show()
    
    return peak_max_w.transpose(), peak_max_h.transpose()

def HS_WFS_preview_grid(pic, peak_w, peak_h, save=False, interpolation=None, 
                        mode='bc', signlevel=0, minlen=3, enhance=False, cmap='gray', 
                        gridweight=0.1, gridcolor='g'):
    """ Preview the grided image based on the peaks detected.
    
    The HS WFS spots should be on the grid cross points.
    
    Keyword arguments:
    pic -- the image matrix.
    peak_w -- peak positions and values of the width (x) direction.
    peak_h -- peak positions and values of the height (y) direction.
    save -- if save the preview. [False]
    interpolation -- the interpolation method that applied to the image, 
        choose 'nearest' to show exactly the experimental data. [None]
    mode -- the grid display mode: ['bc']
        'b': show the background.
        'c': show curved grid.
        's': show simple straight grid.
        'p': show curved grid crossing point.
        combine the 4 modes above to show the grids overlay. (ex: 'bcs', 'csp', etc)
    signlevel -- filter out the trivial peaks that satisfy: [0]
        (peak-avg)/avg < signlevel
    minlen -- minimum length of the valid peak group. [3]
        Ignore the peak group of which the length is shorter than minlen.
    enhance -- if enhance the image to show it clearer. [False]
    cmap -- which color map to use. ['gray']
    gridweight -- the lineweight of the grid lines. [0.1]
    gridcolor -- the color of the grid lines. ['g']
    """
    height, width = pic.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal')
    
    # Background
    if 'b' in mode:
        vmax = 4095 if np.max(pic) > 255 else 255
        if enhance:
            ax.imshow(pic, cmap=cmap, interpolation=interpolation)
        else:
            ax.imshow(pic, vmin=0, vmax=vmax, cmap=cmap, interpolation=interpolation)
    # Simple straight grid
    if 's' in mode:
        for peak_pos in peak_h[0]:
            ax.plot([0, width], [peak_pos, peak_pos], '{}-'.format(gridcolor),
                    lw=gridweight)
        for peak_pos in peak_w[0]:
            ax.plot([peak_pos, peak_pos], [0, height], '{}-'.format(gridcolor),
                    lw=gridweight)
    
    if ('c' in mode) or ('p' in mode):
        X = get_curve_grid(pic, peak_w, peak_h, direction='x', signlevel=signlevel)
        Y = get_curve_grid(pic, peak_w, peak_h, direction='y',
                           signlevel=signlevel).transpose()
        
        # Curved grid
        if 'c' in mode:
            m, n = X.shape
            for i in range(m):
                ax.plot(X[i, :], Y[i, :], '{}-'.format(gridcolor), lw=gridweight)
            for j in range(n):
                ax.plot(X[:, j], Y[:, j], '{}-'.format(gridcolor), lw=gridweight)
                
        # Curved grid points
        if 'p' in mode:
            ax.plot(X, Y, 'g+', markersize=2, mew=0.2)
        
#     tag = '{0}'.format()
#     props = dict(boxstyle='square', facecolor='k', edgecolor='k', alpha=0.2)
#     ax.text(0.95, 0.05, tag, transform=ax.transAxes, fontsize=11, 
#             family='monospace', color='w', va='bottom', ha='right', bbox=props)
    
    fig.tight_layout()
    
    if save:
        fig.savefig('grid_preview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def HS_WFS_get_spot(i, j, pic, peak_pos_w, peak_pos_h, size=20):
    """ Get the spot located at (i th x peak, j th y peak).
    
    Applying the detected peaks positions, get the local image matrix of the
    specific spot.
    
    Keyword arguments:
    i -- the index of the x peaks.
    j -- the index of the y peaks.
    pic -- the image matrix.
    peak_pos_w -- peak positions of the width (x) direction.
    peak_pos_h -- peak positions of the height (y) direction.
    size -- the order of the spot matrix. [20]
    
    Returns:
    spot -- image matrix of the specific spot.
    """
    pos_w = peak_pos_w[i]
    pos_h = peak_pos_h[j]
    a = int(np.ceil(size/2))
    _pos_h_lower = np.max([0, pos_h-a])
    _pos_w_lower = np.max([0, pos_w-a])
    
    spot = np.copy(pic[_pos_h_lower:pos_h+a+1, _pos_w_lower:pos_w+a+1])
    
    return spot

def HS_WFS_filter_spot(spot, signlevel=2, method='gaussian', show=False, save=False, 
                       interpolation=None, enhance=False, cmap='gray'):
    """ Core filter, get the density and position of the spot in the given matrix.
    
    Explain the thoughts (4 steps).
    
    Keyword arguments:
    spot -- the spot matrix.
        note that the x/y order of the matrix should be odd. 
    signlevel -- minimum local signal/background ratio, spot of which s/b ratio is less than
        signlevel would be treated as unvalid spot.
    method -- the filter method. ['gaussian']
    show -- if preview the detected spot. [False]
    interpolation -- the interpolation method that applied to the image, 
        choose 'nearest' to show exactly the experimental data. [None]
    enhance -- if enhance the image to show it clearer. [False]
    cmap -- which color map to use. ['gray']
    
    Returns:
    density, position -- density and position of the detected spot.
    """
    density = 0
    position = [0, 0]
    
    def func(x, A, m, d, C):
        return A*np.exp(-(x-m)**2/(2*d))+C
    
    if method == 'gaussian':
        # Gaussian noise filter
        b, a = np.array(spot.shape)//2
        xw = np.arange(-a, a+1)
        yw = np.mean(spot, 0)
        xh = np.arange(-b, b+1)
        yh = np.mean(spot, 1)
        
        try:
            poptw, pcovw = curve_fit(func, xw, yw, [np.max(yw)-1, 0, (a/2)**2, 1])
            popth, pcovh = curve_fit(func, xh, yh, [np.max(yh)-1, 0, (b/2)**2, 1])

            # Significant filter
            Aw, mw, dw, Cw = poptw
            Ah, mh, dh, Ch = popth
            sigma_w, sigma_h = np.sqrt(dw), np.sqrt(dh)
            C = (Cw+Ch)/2
            A = 1/np.sqrt(2*np.pi)*(Aw*b/sigma_h+Ah*a/sigma_w)
            position = [mw, mh]

            if A/C >= signlevel:
                density = 2*np.pi*A*sigma_w*sigma_h
            else:
                density = 0
        except:
            pass
    elif method == 'fast':
        b, a = np.array(spot.shape)//2
        xw = np.arange(-a, a+1)
        yw = np.mean(spot, 0)
        xh = np.arange(-b, b+1)
        yh = np.mean(spot, 1)
        
        poptw = fast_gaussian_fit(yw, xw)
        popth = fast_gaussian_fit(yh, xh)
        
        # Significant filter
        Aw, mw, sigma_w, Cw = poptw
        Ah, mh, sigma_h, Ch = popth
        C = (Cw+Ch)/2
        A = 1/np.sqrt(2*np.pi)*(Aw*b/sigma_h+Ah*a/sigma_w)
        position = [mw, mh]

        if A/C >= signlevel:
            density = 2*np.pi*A*sigma_w*sigma_h
        else:
            density = 0
    elif method == 'peak':
        b, a = np.array(spot.shape)//2
        xw = np.arange(-a, a+1)
        yw = np.mean(spot, 0)
        xh = np.arange(-b, b+1)
        yh = np.mean(spot, 1)
        
        poptw = peak_gaussian_fit(yw, xw)
        popth = peak_gaussian_fit(yh, xh)
        
        # Significant filter
        Aw, mw, sigma_w, Cw = poptw
        Ah, mh, sigma_h, Ch = popth
        C = (Cw+Ch)/2
        A = 1/np.sqrt(2*np.pi)*(Aw*b/sigma_h+Ah*a/sigma_w)
        position = [mw, mh]

        if A/C >= signlevel:
            density = 2*np.pi*A*sigma_w*sigma_h
        else:
            density = 0        
    else:
        pass
        # Noise filter
        # Salt cleaning
        # Calculate density and position
        # Significant filter
        
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set(xlim=(-a, a), ylim=(b, -b))
        
        vmax = 4095 if np.max(spot) > 255 else 255
        if enhance:
            ax.imshow(spot, cmap=cmap, interpolation=interpolation, 
                      extent=(-a, a, b, -b))
        else:
            ax.imshow(spot, vmin=0, vmax=vmax, cmap=cmap, interpolation=interpolation, 
                      extent=(-a, a, b, -b))
        
        # Normalize the density distribution
        norm = (a+b)/(A+C)
        yw *= norm
        yh *= norm
        
        xw_f = np.arange(-a, a+1, 0.1)
        yw_f = norm*func(xw_f, Aw, mw, sigma_w**2, Cw)
        xh_f = np.arange(-b, b+1, 0.1)
        yh_f = norm*func(xh_f, Ah, mh, sigma_h**2, Ch)

        ax.plot(xw_f, b-yw_f, 'b-')
        ax.plot(a-yh_f, xh_f, 'g-')
        ax.plot(xw, b-yw, 'bs', mec='none')
        ax.plot(a-yh, xh, 'gs', mec='none')
        ax.plot(mw, mh, 'r+', markersize=36, mew=2)
        
        tag = 'density  : {0:.1f}\nposition : ({2:.1f}, {3:.1f})\ns/n ratio: {1:.2f}'.format(density, A/C, *position)
        props = dict(boxstyle='square', facecolor='k', edgecolor='k', alpha=0.2)
        ax.text(0.05, 0.95, tag, transform=ax.transAxes, fontsize=11, family='monospace', color='w', 
                va='top', ha='left', bbox=props)
        
        fig.tight_layout()
        if save:
            fig.savefig('spot.png', dpi=300, bbox_inches='tight')
        plt.show()
    return density, position

def HS_WFS_get_spots(pic, peak_pos_w, peak_pos_h, size=20, signlevel=2, 
                     method='gaussian', show=False, save=False, grid=False, 
                     interpolation=None):
    """ Get the densities and positions of all the detected spots in the given image matrix.
    
    Keyword arguments:
    pic -- the image matrix.
    peak_pos_w -- peak positions of the width (x) direction.
    peak_pos_h -- peak positions of the height (y) direction.
    size -- the order of the spot matrix. [20]
    signlevel -- minimum local signal/background ratio, spot of which s/b ratio is less than
        signlevel would be treated as unvalid spot.
    method -- the filter method. ['gaussian']
    show -- if preview the detected spots. [False]
    save -- if save the preview image. [False]
    interpolation -- the interpolation method that applied to the image, choose 'nearest'
        to show exactly the experimental data. [None]
    
    Returns:
    density, x, y -- density and position matrix of the detected spots.
    """    
    m, n = len(peak_pos_w), len(peak_pos_h)
    density = np.zeros((n, m))
    x = np.zeros((n, m))
    y = np.zeros((n, m))
    
    for i in range(m):
        for j in range(n):
            pos_w, pos_h = peak_pos_w[i], peak_pos_h[j]
            spot = HS_WFS_get_spot(i, j, pic, peak_pos_w, peak_pos_h, size)
            density[j, i], _pos = HS_WFS_filter_spot(spot, signlevel, method)
            x[j, i], y[j, i] = pos_w+_pos[0], pos_h+_pos[1]
        
    if show:
        height, width = pic.shape
    
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.set(xlim=(0, width), ylim=(height, 0))
        # Draw the original image
        vmax = 4095 if np.max(pic) > 255 else 255
        ax.imshow(pic, vmin=0, vmax=vmax, cmap='gray', interpolation=interpolation)
        # Reference grid
        if grid:
            for peak_pos in peak_pos_h:
                ax.plot([0, width], [peak_pos, peak_pos], 'g-', lw=0.1)
            for peak_pos in peak_pos_w:
                ax.plot([peak_pos, peak_pos], [0, height], 'g-', lw=0.1)
        # Mark the detected spots
        _d = density.flatten()
        _x = x.flatten()
        _y = y.flatten()
        
        valid = np.array(_d > 0)
        x_valid = _x[valid]
        y_valid = _y[valid]
        
        invalid = np.array(_d == 0)
        x_invalid = _x[invalid]
        y_invalid = _y[invalid]
        
        ax.plot(x_valid, y_valid, 'g+', ms=2, mew=0.2)
        ax.plot(x_invalid, y_invalid, 'rx', ms=2, mew=0.2)
        
        fig.tight_layout()
        if save:
            fig.savefig('spots_preview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    return density, x, y

def HS_WFS_get_slice_peaks(i, pic, peak_pos_w, peak_pos_h, direction='x', uniform=True, 
                           minlen=3, size=None, lookahead=10, signlevel=0, show=False):
    """ Get peaks of a slice of the HS WFS image.
    
    Project the image to x and y directions, then detect the peaks in both directions.
    
    Keyword arguments:
    i -- the index of the slice.
    pic -- the image matrix.
    peak_pos_w -- peak positions of the width (x) direction.
    peak_pos_h -- peak positions of the height (y) direction.
    direction -- horizontal slice or vertical slice.
    uniform -- if apply the grid uniformor. [True]
    minlen -- minimum length of the valid peak group. [3]
        Ignore the peak group of which the length is shorter than minlen.
    size -- the height of the slice. [None]
        if None, use the height based on the peak_pos.
    lookahead -- how many points to look ahead when detect the peaks. [10]
    signlevel -- filter out the trivial peaks that satisfy: [0]
        (peak-avg)/avg < signlevel
    show -- if show the detected peaks. [False]
    
    Returns:
    peak_max -- peak positions and values of the slice.
    """
    height, width = pic.shape
    # Get slice
    if direction == 'x':
        _peak_pos = peak_pos_h
    else:
        _peak_pos = peak_pos_w
    pos = _peak_pos[i]
    
    if size:
        a = int(np.ceil(size/2))
        i_lower = np.max([0, pos-a])
        if direction == 'x':
            slice = np.copy(pic[i_lower:pos+a+1, :])
        else:
            slice = np.copy(pic[:, i_lower:pos+a+1])
    else:
        i_lower = np.max([0, i-1])
        try:
            pos_lower = _peak_pos[i_lower]
            pos_upper = _peak_pos[i+1]
        except IndexError:
            pos_lower = _peak_pos[i_lower]
            pos_upper = 2*pos-pos_lower
        if direction == 'x':
            slice = np.copy(pic[pos_lower:pos_upper+1, :])
        else:
            slice = np.copy(pic[:, pos_lower:pos_upper+1])

    if direction == 'x':
        x = np.arange(width)
        y = np.mean(slice, 0)
    else:
        x = np.arange(height)
        y = np.mean(slice, 1)
    # Peak detect
    peak_max, peak_min = peakdetect(y, x, lookahead, )
    peak_max = np.array(peak_max)
    # Filter out the trivial peaks
#     def signfilter(i):
#         if i == 0:
#             x0 = x[0]
#             x1, x2 = peak_max[0:2, 0]
#         else:
#             try:
#                 x0, x1, x2 = peak_max[i-1:i+2, 0]
#             except ValueError:
#                 x0, x1 = peak_max[i-1:i+1, 0]
#                 x2 = x[-1]
#         xlower, xupper = (x0+x1)/2, (x1+x2)/2
#         index = (x <= xupper) & (x >= xlower)
#         avg = np.mean(y[index])
#         peak = peak_max[i, 1]

#         if (peak-avg)/avg >= signlevel:
#             return True
#         else:
#             return False
    def signfilter(i, r=20):
        if i == 0:
            x0 = x[0]
            x1, x2 = peak_max[0:2, 0]
        else:
            try:
                x0, x1, x2 = peak_max[i-1:i+2, 0]
            except ValueError:
                x0, x1 = peak_max[i-1:i+1, 0]
                x2 = x[-1]
        x0 = x1-np.min([r, x1-x0])
        x2 = x1+np.min([r, x2-x1])
        
        xlower, xupper = (x0+x1)/2, (x1+x2)/2
        index = (x <= xupper) & (x >= xlower)
        yn = y[index]
        avg = (yn[0]+yn[-1])/2
        peak = peak_max[i, 1]

        if (peak-avg)/avg >= signlevel:
            return True
        else:
            return False
        
    valid = np.array([signfilter(p) for p in range(peak_max.shape[0])])
    peak_max = peak_max[valid, :]
    
    if uniform:
        peak_max = grid_uniformor(peak_max, minlen=minlen)
        
    peak_pos = peak_max[:, :1]
    peak_value = peak_max[:, 1:]

    if show:
        xlim = width if direction == 'x' else height
        dirtag = 'horizontal' if direction == 'x' else 'vertical'
        color = 'b' if direction == 'x' else 'g'
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.set(xlabel='pixel number (1)', ylabel='projected density (arb. units.)', 
               xlim=(0, xlim))

        ax.plot(x, y, '{}-'.format(color), 
                label='{3} slice {2}\nlookahead = {0}\nsignlevel = {1}'.format(lookahead, signlevel, i, dirtag))
        if len(peak_pos):
            ax.plot(peak_pos, peak_value, 'rx')

        ax.legend(loc=0)
        fig.tight_layout()
        # fig.savefig('peaks.pdf', bbox_inches='tight')
        plt.show()
        
    return peak_max.transpose()