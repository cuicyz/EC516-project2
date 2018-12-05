import scipy.io
import numpy as np
import matplotlib
import math
from numpy import fft
from scipy.io import wavfile
from matplotlib import pyplot as plt

#Read the signal
sample_rate, original_signal = wavfile.read('project2.wav')
original_signal=(original_signal[:,0]/2+original_signal[:,1]/2)
time = np.arange(len(original_signal))/float(sample_rate)
print("The sample rate is:",sample_rate)
print("The length of the signal is:",len(original_signal))
print("The time of the signal is:",time[-1])

#Make windowed signal
window_length=int(0.02*sample_rate)
start_time=int(0.5*sample_rate)
windowed_signal=original_signal[start_time:start_time+window_length]
windowed_time=time[start_time:start_time+window_length]

#Display the original signal
#plt.plot(time, original_signal)
#plt.title('Speech signal')
#plt.xlabel('Time')
#plt.ylabel('Amplitude')
#plt.savefig('Original_signal.jpg')

#Display the windowed signal
#plt.plot(windowed_time, windowed_signal)
#plt.title('Windowed interval signal')
#plt.xlabel('Time')
#plt.ylabel('Amplitude')
#plt.savefig('Windowed_signal.jpg')

#Make dft of windowed signal
DFT=np.fft.fft(windowed_signal)
print(len(DFT))
DFT_freq=np.arange(0,sample_rate,50)
half_DFT_freq=DFT_freq[0:int(len(DFT_freq)/2)]
half_DFT=abs(DFT)[0:int(len(DFT_freq)/2)]

#Display dft of windowed signal
#plt.plot(half_DFT_freq, half_DFT)
#plt.title('DFT of windowed interval signal')
#plt.yscale("log")
#plt.xlabel('Frequency')
#plt.ylabel('Log of Amplitude')
#plt.savefig('DFT of Windowed_interval_signal.jpg')

#Making rn
r0=0;r1=0;r2=0;r3=0;r4=0;r5=0;r6=0;r7=0;r8=0;r9=0;r10=0
windowed_signal_add=np.append(windowed_signal,[0,0,0,0,0,0,0,0,0,0])
for i in range(0,len(windowed_signal)):
    r0+=windowed_signal_add[i]**2
    r1+=windowed_signal_add[i]*windowed_signal_add[i+1]
    r2+=windowed_signal_add[i]*windowed_signal_add[i+2]
    r3+=windowed_signal_add[i]*windowed_signal_add[i+3]
    r4+=windowed_signal_add[i]*windowed_signal_add[i+4]
    r5+=windowed_signal_add[i]*windowed_signal_add[i+5]
    r6+=windowed_signal_add[i]*windowed_signal_add[i+6]
    r7+=windowed_signal_add[i]*windowed_signal_add[i+7]
    r8+=windowed_signal_add[i]*windowed_signal_add[i+8]
    r9+=windowed_signal_add[i]*windowed_signal_add[i+9]
    r10+=windowed_signal_add[i]*windowed_signal_add[i+10]
print('r0:',r0)
print('r1:',r1)
print('r2:',r2)
print('r3:',r3)
print('r4:',r4)
print('r5:',r5)
print('r6:',r6)
print('r7:',r7)
print('r8:',r8)
print('r9:',r9)
print('r10:',r10)

#Solving a_k
r=np.array([[r0,r1,r2,r3,r4,r5,r6,r7,r8,r9],[r1,r0,r1,r2,r3,r4,r5,r6,r7,r8],[r2,r1,r0,r1,r2,r3,r4,r5,r6,r7],[r3,r2,r1,r0,r1,r2,r3,r4,r5,r6],[r4,r3,r2,r1,r0,r1,r2,r3,r4,r5],[r5,r4,r3,r2,r1,r0,r1,r2,r3,r4],[r6,r5,r4,r3,r2,r1,r0,r1,r2,r3],[r7,r6,r5,r4,r3,r2,r1,r0,r1,r2],[r8,r7,r6,r5,r4,r3,r2,r1,r0,r1],[r9,r8,r7,r6,r5,r4,r3,r2,r1,r0]])
r_inverse=np.linalg.inv(r)
fix_matrix=np.array([[-r1],[-r2],[-r3],[-r4],[-r5],[-r6],[-r7],[-r8],[-r9],[-r10]])
a_matrix=r_inverse@fix_matrix
a=[0]*10
for i in range(len(a_matrix)):
    a[i]=a_matrix[i][0]
    print(a[i])

#Solving G
G_square=r0+r1*a[0]+r2*a[1]+r3*a[2]+r4*a[3]+r5*a[4]+r6*a[5]+r7*a[6]+r8*a[7]+r9*a[8]+r10*a[9]
G=G_square**0.5
print('G:',G)

#Making x[] and H[]
point=11
x=[0]*point
FFT_x=[0]*point
x[0]=1/G
for i in range(1,11):
    x[i]=a[i-1]/G
print(x)
FFT_x=np.fft.fft(x)
print(FFT_x)
H=[0]*point
for i in range(11):
    H[i]=1/FFT_x[i]
    H[i]=np.absolute(H[i])
print(H)

#Display H(z)
plt.plot(np.arange(point), H)
plt.title('H(z)')
plt.yscale("log")
plt.xlabel('n')
plt.ylabel('log of H(z)')
plt.savefig('H(z).jpg')
        


