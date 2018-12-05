import scipy.io
import numpy as np
import matplotlib
import math
from numpy import fft
from scipy.io import wavfile
from matplotlib import pyplot as plt

#Read the signal
sample_rate, original_signal = wavfile.read('dsp.wav')
time = np.arange(len(original_signal))/float(sample_rate)
print("The sample rate is:",sample_rate)
print("The length of the signal is:",len(original_signal))
print("The time of the signal is:",time[-1])
window_length=int(0.02*sample_rate)
start_time=int(0.7*sample_rate)
windowed_signal=original_signal[start_time:start_time+window_length]
windowed_time=time[start_time:start_time+window_length]
print(windowed_signal)
print(original_signal)


#Display the signal
#plt.plot(time, original_signal)
#plt.title('Speech signal')
#plt.xlabel('Time')
#plt.ylabel('Amplitude')
#plt.savefig('Original_signal.jpg')

#plt.plot(windowed_time, windowed_signal)
#plt.title('Windowed interval signal')
#plt.xlabel('Time')
#plt.ylabel('Amplitude')
#plt.savefig('Windowed_interval_signal.jpg')

DFT=np.fft.fft(windowed_signal)
DFT_time=np.arange(len(windowed_signal))
half_DFT_time=DFT_time[0:int(len(DFT_time)/2)]
half_DFT=abs(DFT)[0:int(len(DFT_time)/2)]
log_half_DFT=[0]*len(half_DFT)
for i in range(len(half_DFT)):
    log_half_DFT[i]=math.log(half_DFT[i])

plt.plot(half_DFT_time, log_half_DFT)
plt.title('DFT of windowed interval signal')
plt.xlabel('Time')
plt.ylabel('Log of Amplitude')
plt.savefig('DFT of Windowed_interval_signal.jpg')

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

r=np.array([[r0,r1,r2,r3,r4,r5,r6,r7,r8,r9],[r1,r0,r1,r2,r3,r4,r5,r6,r7,r8],[r2,r1,r0,r1,r2,r3,r4,r5,r6,r7],[r3,r2,r1,r0,r1,r2,r3,r4,r5,r6],[r4,r3,r2,r1,r0,r1,r2,r3,r4,r5],[r5,r4,r3,r2,r1,r0,r1,r2,r3,r4],[r6,r5,r4,r3,r2,r1,r0,r1,r2,r3],[r7,r6,r5,r4,r3,r2,r1,r0,r1,r2],[r8,r7,r6,r5,r4,r3,r2,r1,r0,r1],[r9,r8,r7,r6,r5,r4,r3,r2,r1,r0]])
r_inverse=np.linalg.inv(r)
fix_matrix=np.array([[-r1],[-r2],[-r3],[-r4],[-r5],[-r6],[-r7],[-r8],[-r9],[-r10]])
a_matrix=r_inverse@fix_matrix
a=[0]*10
for i in range(len(a_matrix)):
    a[i]=a_matrix[i]
    print(a[i])

G_square=r0+r1*a[0]+r2*a[1]+r3*a[2]+r4*a[3]+r5*a[4]+r6*a[5]+r7*a[6]+r8*a[7]+r9*a[8]+r10*a[9]
G=G_square**0.5
print(G)

x=[0]*11
x[0]=1/G
for i in range(1,11):
    x[i]=a[i-1]/G
FFT_x=np.fft.fft(x)
H=[0]*11
for i in range(11):
    H[i]=1/x[i]
print(H)

plt.plot(np.arange(11), H)
plt.title('H(z)')
plt.xlabel('Time')
plt.ylabel('H(z)')
plt.savefig('H(z).jpg')
        


