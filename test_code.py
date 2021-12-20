import spikertools as sp
import matplotlib.pyplot as plt

#importing session
s1 = sp.Session(r"C:\Users\USER\Documents\BYB\BYB_Recording_2021-06-18_16.14.32.wav")

#plot overview without events
#s1.plot_overview(show_events=False)

#plot overview with events
s1.plot_overview()

