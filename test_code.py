import spikertools as sp
import matplotlib.pyplot as plt

#importing session
s1 = sp.Session(r"C:\Users\USER\Documents\BYB\BYB_Recording_2021-06-18_16.14.32.wav")

s2 = sp.Session(r"C:\Users\USER\Downloads\Copy of full_trial_ari.wav")

#plot overview without events
#s1.plot_overview(show_events=False)

#plot overview with events
#s2.plot_overview()


#plot interval (one channel) without events or legend
#s1.plot_interval(0,(0,40))

#plot interval (one channel) with events, without legend
#s1.plot_interval(0, (0,40), events=True)

#plot interval (one channel) with events, with legend
#s1.plot_interval(0, (0,40), events=True, legends=True)



#plot interval (more than one channel) without events or legend
#s1.plot_interval([0,1],(0,40))

#plot interval (more than one channel) with events, without legend
#s1.plot_interval([0,1], (0,40), events=True)

#plot interval (more than one channel) with events, with legend
#s1.plot_interval([0,1], (0,40), events=True, legends=True)

#pile plot
#s1.pileplot("3", (2,2))

#time locked average plot without traces
#1.tlavgplot("3", (2,2))

#time locked average plot with traces
#s1.tlavgplot("3", (2,2), showtraces= True)

#joydiv plot
#s1.joydivplot("3", (2,2))
s2.plot_joydiv("6", (-1,3),spec_channel=1)
#raster plot
#s1.rasterplot(0)


# TESTING COLOR CHANGE

#s1.get_channel(0).set_color("r")
#s1.plot_interval(0,(0,40))

#s2.plot_overview()