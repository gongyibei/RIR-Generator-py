from math import *
import numpy as np
import pickle 


def sinc(x):
    if x == 0: 
        return 1.0
    else:
        return sin(x)/x

def sim_microphone(x, y, z, angle, mtype):

    #  Polar Pattern         alpha
    #  ---------------------------
    #  Bidirectional         0
    #  Hypercardioid         0.25    
    #  Cardioid              0.5
    #  Subcardioid           0.75
    #  Omnidirectional       1

    if mtype in 'bcsh':
        if mtype == 'b':
            rho = 0
        elif mtype == 'h':
            rho = 0.25
        elif mtype == 'c':
            rho = 0.5
        elif mtype == 's':
            rho = 0.75
        else:
            rho = 1


        print(x, y, z)
        vartheta = acos(z/sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)))
        varphi = atan2(y, x)
        gain = sin(pi/2-angle[1]) * sin(vartheta) * cos(angle[0]-varphi) + cos(pi/2-angle[1]) * cos(vartheta)
        gain = rho + (1-rho) * gain
        return gain

    else:
        return 1


def computeRIR(c, fs, rr, nMicrophones, nSamples, ss, LL, beta, microphone_type, nOrder, angle, isHighPassFilter):
    imp = np.zeros([nSamples, nMicrophones], dtype=np.double)
    W = 2*pi*100/fs
    R1 = exp(-W)
    B1 = 2*R1*cos(W)
    B2 = -R1 * R1
    A1 = -(1+R1)
    
    Fc = 1
    Tw = 2 * round(0.004*fs)
    cTs = c/fs

    LPI = [0]*Tw
    r = [0] * 3
    s = [0] * 3
    L = [0] * 3
    Rm = [0] * 3
    Rp_plus_Rm = [0] * 3
    refl = [0] * 3

    s[0] = ss[0]/cTs
    s[1] = ss[1]/cTs
    s[2] = ss[2]/cTs
    L[0] = LL[0]/cTs
    L[1] = LL[1]/cTs
    L[2] = LL[2]/cTs

    for idxMicrophone in range(nMicrophones):
        # [x_1 x_2 ... x_N y_1 y_2 ... y_N z_1 z_2 ... z_N]
        r[0] = rr[idxMicrophone + 0*nMicrophones] / cTs
        r[1] = rr[idxMicrophone + 1*nMicrophones] / cTs
        r[2] = rr[idxMicrophone + 2*nMicrophones] / cTs

        n1 = int(ceil(nSamples/(2*L[0])))
        n2 = int(ceil(nSamples/(2*L[1])))
        n3 = int(ceil(nSamples/(2*L[2])))

        # Generate room impulse response
        for mx in range(-n1, n1 + 1, 1):
            Rm[0] = 2*mx*L[0]

            for my in range(-n2, n2 + 1, 1):
                Rm[1] = 2*my*L[1]

                for mz in range(-n3, n3 + 1, 1):
                    Rm[2] = 2*mz*L[2]

                    for q in range(2):
                        Rp_plus_Rm[0] = (1-2*q)*s[0] - r[0] + Rm[0]
                        print(beta, mx, q)
                        refl[0] = pow(beta[0], abs(mx-q)) * pow(beta[1], abs(mx))
                        
                        for j in range(2):
                            Rp_plus_Rm[1] = (1-2*j)*s[1] - r[1] + Rm[1]
                            refl[1] = pow(beta[2], abs(my-j)) * pow(beta[3], abs(my))

                            for k in range(2):

                                Rp_plus_Rm[2] = (1-2*k)*s[2] - r[2] + Rm[2]
                                refl[2] = pow(beta[4],abs(mz-k)) * pow(beta[5], abs(mz))

                                dist = sqrt(pow(Rp_plus_Rm[0], 2) + pow(Rp_plus_Rm[1], 2) + pow(Rp_plus_Rm[2], 2))

                                if abs(2*mx-q)+abs(2*my-j)+abs(2*mz-k) <= nOrder or nOrder == -1:
                                    fdist = floor(dist)

                                    if fdist < nSamples:
                                        gain = sim_microphone(Rp_plus_Rm[0], Rp_plus_Rm[1], Rp_plus_Rm[2], angle, microphone_type[0]) * refl[0]*refl[1]*refl[2]/(4*pi*dist*cTs)

                                        for n in range(Tw):
                                            LPI[n] =  0.5 * (1 - cos(2*pi*((n+1-(dist-fdist))/Tw))) * Fc * sinc(pi*Fc*(n+1-(dist-fdist)-(Tw/2)))

                                        startPosition = int(fdist-(Tw/2)+1)

                                        for n in range(Tw):
                                            if startPosition+n >= 0 and startPosition+n < nSamples:
                                                imp[idxMicrophone + nMicrophones*(startPosition+n)] += gain * LPI[n]

        #  'Original' high-pass filter as proposed by Allen and Berkley.
        if isHighPassFilter == 1 :
            Y = [0] * 3

            for idx in range(nSamples):

                X0 = imp[idx][idxMicrophone]
                Y[2] = Y[1]
                Y[1] = Y[0]
                Y[0] = B1*Y[1] + B2*Y[2] + X0
                imp[idx][idxMicrophone] = Y[0] + A1*Y[1] + R1*Y[2]
    return imp

def rir_generator(c,samplingRate,micPositions,srcPosition,LL,**kwargs):

	if type(LL) is not np.array:
		LL=np.array(LL,ndmin=2)
	if LL.shape[0]==1:
		LL=np.transpose(LL)

	if type(micPositions) is not np.array:
		micPositions=np.array(micPositions,ndmin=2)
	if type(srcPosition) is not np.array:
		srcPosition=np.array(srcPosition,ndmin=2)

	"""Passing beta"""
	beta = np.zeros([6,1], dtype=np.double)
	if 'beta' in kwargs:
		betaIn=kwargs['beta']
		if type(betaIn) is not np.array:
			betaIn=np.transpose(np.array(betaIn,ndmin=2))
		if (betaIn.shape[1])>1:
			beta=betaIn
			V=LL[0]*LL[1]*LL[2]
			alpha = ((1-pow(beta[0],2))+(1-pow(beta[1],2)))*LL[0]*LL[2]+((1-pow(beta[2],2))+(1-pow(beta[3],2)))*LL[1]*LL[2]+((1-pow(beta[4],2))+(1-pow(beta[5],2)))*LL[0]*LL[1]
			reverberation_time = 24*np.log(10.0)*V/(c*alpha)
			if (reverberation_time < 0.128):
				reverberation_time = 0.128
		else:
			reverberation_time=betaIn		
			if (reverberation_time != 0) :
				V=LL[0]*LL[1]*LL[2]
				S = 2*(LL[0]*LL[2]+LL[1]*LL[2]+LL[0]*LL[1])		
				alfa = 24*V*np.log(10.0)/(c*S*reverberation_time)
				if alfa>1:
					raise ValueError("Error: The reflection coefficients cannot be calculated using the current room parameters, i.e. room size and reverberation time.\n Please specify the reflection coefficients or change the room parameters.")
				beta=np.zeros([6,1])
				beta+=np.sqrt(1-alfa)
			else:
				beta=np.zeros([6,1])
	else:
			raise ValueError("Error: Specify either RT60 (ex: beta=0.4) or reflection coefficients (beta=[0.3,0.2,0.5,0.1,0.1,0.1])")
	
	"""Number of samples: Default T60 * Fs"""
	if 'nsample' in kwargs:
		nsamples=kwargs['nsample']
	else:
		nsamples=int(reverberation_time * samplingRate)

	"""Mic type: Default omnidirectional"""
	m_type='omnidirectional'
	if 'mtype' in kwargs:
		m_type=kwargs['mtype']
	if m_type is 'bidirectional':
		mtype = 'b'
	if m_type is 'cardioid':
		mtype = 'c'
	if m_type is 'subcardioid':
		mtype = 's'
	if m_type is 'hypercardioid':
		mtype = 'h'
	if m_type is 'omnidirectional':
		mtype = 'o'		

	"""Reflection order: Default -1"""
	order = -1
	if 'order' in kwargs:
		order = kwargs['order']
		if order<-1:
			raise ValueError("Invalid input: reflection order should be > -1")

	"""Room dimensions: Default 3"""
	dim=3
	if 'dim' in kwargs:
		dim=kwargs['dim']
		if dim not in [2,3]:
			raise ValueError("Invalid input: 2 or 3 dimensions expected")
		if dim is 2:
			beta[4]=0
			beta[5]=0

	"""Orientation"""
	angle = np.zeros([2,1], dtype=np.double)
	if 'orientation' in kwargs:
		orientation=kwargs['orientation']
		if type(orientation) is not np.array:
			orientation=np.array(orientation,ndmin=2)
		if orientation.shape[1]==1:
			angle[0]=orientation[0]
		else:
			angle[0]=orientation[0,0]

			angle[1]=orientation[0,1]

	"""hp_filter enable"""
	isHighPassFilter=1
	if 'hp_filter' in kwargs:
		isHighPassFilter=kwargs['hp_filter']


	numMics=micPositions.shape[0]

	"""Create output vector"""
	imp = np.zeros([nsamples,numMics], dtype=np.double)

	roomDim = np.ascontiguousarray(LL.astype('double'), dtype=np.double)
	micPos = np.ascontiguousarray(np.transpose(micPositions).astype('double'), dtype=np.double)	
	srcPos = np.ascontiguousarray(np.transpose(srcPosition).astype('double'), dtype=np.double)	

	imp = computeRIR(c, samplingRate, micPos, numMics, nsamples, srcPos, roomDim, beta, mtype, order, angle, isHighPassFilter)
	return imp.T[0]
