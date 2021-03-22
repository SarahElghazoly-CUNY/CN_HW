# April 24, 1010

import numpy as np
import scipy.special

def plpva(x, xmin, vec=np.arange(1.50, 3.51, 0.01), reps=1000, quiet=False, **kwargs):
    """
% PLPVA calculates the p-value for the given power-law fit to some data.
%    Source: http://www.santafe.edu/~aaronc/powerlaws/
% 
%    PLPVA(x, xmin) takes data x and given lower cutoff for the power-law
%    behavior xmin and computes the corresponding p-value for the
%    Kolmogorov-Smirnov test, according to the method described in 
%    Clauset, Shalizi, Newman (2007).
%    PLPVA automatically detects whether x is composed of real or integer
%    values, and applies the appropriate method. For discrete data, if
%    min(x) > 1000, PLPVA uses the continuous approximation, which is 
%    a reliable in this regime.
%   
%    The fitting procedure works as follows:
%    1) For each possible choice of x_min, we estimate alpha via the 
%       method of maximum likelihood, and calculate the Kolmogorov-Smirnov
%       goodness-of-fit (gof) statistic D.
%    2) We then select as our estimate of x_min, the value that gives the
%       minimum value D over all values of x_min.
%
%    Note that this procedure gives no estimate of the uncertainty of the 
%    fitted parameters, nor of the validity of the fit.
%
%    Example:
%       x = (1-np.random.random(10000))**(-1/(2.5-1))
%       p, gof = plpva.plpva(x, 1)
% 
% Notes:
% 
% 1. In order to implement the integer-based methods, the numeric
%    maximization of the log-likelihood function was used. This requires
%    that we specify the range of scaling parameters considered. We set
%    this range to be np.arange(1.50, 3.51, 0.01) by default. This vector
%    can be set by the user like so,
%    
%       p, gof = plpva.plpva(x, 1, vec=np.arange(1.001, 5.002, 0.001))
%
% 2. PLFIT can be told to limit the range of values considered as estimates
%    for xmin in two ways. First, it can be instructed to sample these
%    possible values like so,
%    
%       p, gof = plpva.plpva(x, 1, sample=100)
%    
%    which uses 100 uniformly distributed values on the sorted list of
%    unique values in the data set. Alternatively, it can simply omit all
%    candidates above a hard limit, like so
%    
%       p, gof = plpva.plpva(x, 1, limit=3.4)
%    
%    In the case of discrete data, it rounds the limit to the nearest
%    integer.
% 
% 3. The default number of semiparametric repetitions of the fitting
% procedure is 1000. This number can be changed like so
%    
%       p, gof = plpva.plpva(x, 1, reps=10000)
% 
    """

    x = np.array(x)

    N = int(len(x))

    nof = np.array([])

    if not quiet:
        print('Power-law Distribution, p-value calculation')
        print('   Warning: This can be a slow calculation; please be patient.')
        print('   n    = %i\n   xmin = %6.4f\n   reps = %i' % (len(x), xmin, reps))

    # continuous method
    if (min(x) > 1000 and len(x) > 100) or sum(x-np.floor(x)):
        # compute D for the empirical distribution
        z     = x[x >= xmin]
        nz    = float(len(z))
        y     = x[x < xmin]
        ny    = float(len(y))
        alpha = 1 + nz/sum(np.log(z/xmin))
        cz    = np.arange(nz)/nz
        cf    = 1 - (xmin/np.sort(z))**(alpha-1)
        gof   = max(abs( cz - cf ))
        pz    = nz/N
        
        # compute distribution of gofs from semi-parametric bootstrap
        # of entire data set with fit
        for B in range(reps):
            # semi-parametric bootstrap of data
            n1 = sum(np.random.random(int(N)) > pz)
            q1 = y[np.array(np.floor(ny*np.random.random(int(n1))), dtype='int32')]
            n2 = N - n1
            q2 = xmin*(1-np.random.random(n2))**(-1/(alpha-1))
            q  = np.sort(np.r_[q1, q2])

            # estimate xmin and alpha via GoF-method
            qmins = np.unique(q)
            qmins = qmins[0:-1]
            try:
                qmins = qmins[qmins <= kwargs['limit']]
            except KeyError:
                pass
            try:
                qmins = qmins[np.array(np.unique(np.round(np.linspace(1, len(qmins), kwargs['sample']) - 1)), dtype='int32')]
            except KeyError:
                pass
            dat = np.array([])
            for qmin in qmins:
                zq  = q[q >= qmin]
                nq  = float(len(zq))
                a   = nq / sum(np.log(zq/qmin))
                cq  = np.arange(nq) / nq
                cf  = 1 - (qmin/zq)**a
                dat = np.r_[dat, max(abs( cq - cf ))]
            # store distribution of estimated gof values
            nof = np.r_[nof, min(dat)]
            if not quiet:
#                fprintf('[%i]\tp = %6.4f\t[%4.2fm]\n',B,sum(nof(1:B)>=gof)./B,toc/60);
#                print('[%i]\tp = %6.4f\t[%4.2fm]\n' % (B+1, sum(nof>=gof)/float(B+1), toc/60))
                print('[%i]\tp = %.4f\n' % (B+1, sum(nof>=gof) / float(B+1)))
        p = sum(nof>=gof) / float(len(nof))

    # discrete method
    else:
        zvec  = scipy.special.zeta(vec, xmin) # (2.5)

        # compute D for the empirical distribution
        z     = x[x >= xmin]
        nz    = float(len(z))
        xmax  = max(z)
        y     = x[x < xmin]
        ny    = float(len(y))

        L     = - np.inf * np.ones(len(vec))
        slogz = sum(np.log(z))
        for k in range(len(vec)):
            L[k] = - nz*np.log(zvec[k]) - vec[k]*slogz # (3.5) (B.8)
        Y, I  = L.max(0), L.argmax(0)
        alpha = vec[I]

        fit = np.cumsum((np.arange(xmin, xmax+1)**-alpha) / zvec[I]) # P(x)
        # cdi = np.cumsum(np.histogram(z, np.arange(xmin, xmax+2), new=True)[0] / nz) # S(x)
        cdi = np.cumsum(np.histogram(z, np.arange(xmin, xmax+2))[0] / nz) # S(x)
        gof = max(abs( fit - cdi )) # (3.9)
        pz  = nz/N

        mmax = int(20*xmax)
        

        
        pdf = np.r_[np.zeros(int(xmin-1)), (np.arange(xmin, mmax+1)**int(alpha)) / zvec[I]]
        cdf = np.r_[[np.arange(1, int(mmax+2))], [np.r_[np.cumsum(pdf), 1]]]

        # compute distribution of gofs from semi-parametric bootstrap
        # of entire data set with fit
        for B in range(reps):
            # semi-parametric bootstrap of data
            n1 = sum(np.random.random(int(N)) > pz)
            q1 = y[np.array(np.floor(ny*np.random.random(int(n1))), dtype='int32')]
            n2 = N - n1

            # simple discrete zeta generator
            r2 = np.sort(np.random.random(n2))
            c  = 0
            q2 = np.zeros(n2)
            k  = 0
            for i in range(int(xmin), int(mmax+2)):
               while c<len(r2) and r2[c]<=cdf[1,i-1]:
                   c=c+1
               q2[k:c] = i
               k = c
               if k >= n2:
                   break
            q = np.r_[q1, q2]

            # estimate xmin and alpha via GoF-method
            qmins = np.unique(q)
            qmins = qmins[0:-1]
            try:
                qmins = qmins[qmins <= kwargs['limit']]
            except KeyError:
                pass
            try:
                qmins = qmins[np.array(np.unique(np.round(np.linspace(1, len(qmins), kwargs['sample']) - 1)), dtype='int32')]
            except KeyError:
                pass
            dat  = np.array([])
            qmax = max(q)
            zq   = q
            for qmin in qmins:
                zq     = zq[zq >= qmin]
                nq     = float(len(zq))
                slogzq = sum(np.log(zq))
                if nq > 1:
                    try:
                        # vectorized version of numerical calculation
                        L =  - nq*np.log(zvec)- vec*slogzq # (3.5) (B.8)
                    except:
                        # iterative version (more memory efficient, but slower)
                        print("except")
                        L = - np.inf * np.ones(len(vec))
                        for k in range(len(vec)):
                            L[k] = - nq*np.log(zvec[k]) - vec[k]*slogzq # (3.5) (B.8)
                    Y, I = L.max(0), L.argmax(0)
                
                    fit = np.cumsum((np.arange(qmin, qmax+1)**-vec[I]) / zvec[I]) # P(x)
                    # cdi = np.cumsum(np.histogram(zq, np.arange(qmin, qmax+2), new=True)[0] / nq) # S(x)
                    cdi = np.cumsum(np.histogram(zq, np.arange(qmin, qmax+2))[0] / nq) # S(x)
                    dat = np.r_[dat, max(abs( fit - cdi ))] # (3.9)
                else:
                    dat = np.r_[dat, -np.inf]

            # -- store distribution of estimated gof values
            nof = np.r_[nof, min(dat)]
            if not quiet:
#                fprintf('[%i]\tp = %6.4f\t[%4.2fm]\n',B,sum(nof(1:B)>=gof)./B,toc/60);
#                print('[%i]\tp = %6.4f\t[%4.2fm]\n' % (B+1, sum(nof>=gof)/float(B+1), toc/60))
                print('[%i]\tp = %.4f\n' % (B+1, sum(nof>=gof) / float(B+1)))
        p = sum(nof>=gof) / float(len(nof))

    return p, gof
