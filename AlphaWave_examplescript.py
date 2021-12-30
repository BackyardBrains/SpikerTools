# this is an empty script used for testing, kindly disregard

import json
import spikertools as sp
from matplotlib import pyplot as plt
#peth_test = sp.Session("dcmd_example_data\BYB_Recording_2016-07-29_16.29.37.wav", "")
#with open("C:/Users/USER\Desktop/Experiment0708A.json") as f:
#    data = json.load(f)
#    print(data)

#peth_test.plot_peth(0, (-1,1), "2", "1", 20)
#test_vec = [4]*5
#plt.scatter([0,2,3,4,5], test_vec)
#plt.show()
#peth_test.plot_overview(show_events=False)

#s1 = sp.Session("alphawave_example_data\BYB_Recording_2015-07-26_21.47.25.wav")
#s2 = sp.Session("p300_example_data\BYB_Recording_2019-06-11_13.23.58.wav")

#s1.set_sessionID("alpha")
#s2.set_sessionID("p300")

#exp = sp.Sessions([s1,s2])

#exp.plot_overview()

#exp.plot_interval(0, (0,1), events=True, legends= True, join=False)

#exp.plot_psd(0,(0,10), (0,20), join=True)

peth1 = sp.Session()
peth2 = sp.Session()

import json
with open("dcmd_example_data\G27-072516-03\G27-072516-03.json") as f:
    data = json.load(f)
    trials = data["trials"]
    onsets1 = []
    spikes1 = []
    i=0
    for trial in trials:
        onset = trial["timeOfImpact"] + 10*i
        spike_data = trial["spikeTimestamps"]
        for spike in spike_data:
            spikes1.append(spike + 10*i)
        i=i+1
        onsets1.append(onset)

with open("dcmd_example_data\G25-072416-01\G25-072416-01.json") as f:
    data = json.load(f)
    trials = data["trials"]
    onsets2 = []
    spikes2 = []
    i=0
    for trial in trials:
        onset = trial["timeOfImpact"] + 10*i
        spike_data = trial["spikeTimestamps"]
        for spike in spike_data:
            spikes2.append(spike + 10*i)
        i=i+1
        onsets2.append(onset)

events1 = {}
events1["onsets"] = onsets1
events1["spikes"] = spikes1

events2 = {}
events2["onsets"] = onsets2
events2["spikes"] = spikes2

peth1.set_events(events1)
peth2.set_events(events2)
peth1.set_sessionID("peth 1")
peth2.set_sessionID("peth 2")


peths = sp.Sessions([peth1,peth2])


peths.plot_peth(0, (-1,1), 'onsets', 'spikes', 50)
