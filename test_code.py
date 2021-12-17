import spikertools as sp
import matplotlib.pyplot as plt

#importing session
s1 = sp.Session(r"C:\Users\USER\Documents\BYB\BYB_Recording_2021-06-18_16.14.32.wav")

#testing plot interval function
#s1.plot_interval([0,1],1,200, events=True, make_fig=False)


#testing plot overview function
#s1.plot_overview()

s1.plot_overview(show_events=True)

#s1.plot_overview(show_events=True, show_legends=True)
