#!/usr/bin/python

import numpy as np

######### SIMULATION PARAMETERS ########

R_NP = 55.
dr = 0.1

N = 1024
L = 200.
V = L*L*L - 4 * 3.1416 * R_NP*R_NP*R_NP / 3

rho = N/V

########################################



Full_rdf = open("rdf.dat","r")
#HC_rdf = open("rdf_HC.dat","r")

M_Full_rdf = np.loadtxt(Full_rdf)
#M_HC_rdf = np.loadtxt(HC_rdf)

Full_rdf.close()
#HC_rdf.close()

# calc the tail mean

npoints = len(M_Full_rdf) - 1

start = (int)(npoints*0.95)

mean = 0.
count = 0

for i in range(start,npoints,1):
	mean += M_Full_rdf[i][1]
	count += 1

mean /= count

mean_Full_rdf = mean



print mean_Full_rdf


R = 3.72
R = 6.58

# Count the proteins by integration of the HC

nprot = 0.

for i in range(npoints):
	r = i*dr

	if(r < 1.5*R):
		nprot += 4 * 3.1416 * (R_NP + r)*(R_NP + r) * dr * M_Full_rdf[i][1] * rho

nprot_HC = (int)(nprot)

print "Proteins in the HC %d" % (nprot_HC)
print "The fraction bound of the HC is %1.2f" % (nprot_HC*1./N)


# Integrate the SC to count the proteins

nprot = 0.

for i in range(npoints):
	
	r = i*dr
	
	if r > 1.5*R:
		rdf = M_Full_rdf[i][1]
		
		if r > 3.5*R:
			rdf -= mean_Full_rdf
		
		nprot += 4 * 3.1416 * (R_NP + r)*(R_NP + r) * dr * (rdf) * rho

nprot_SC = (int)(nprot)

print "Proteins in the SC %d" % (nprot_SC)
print "The fraction bound of the SC is %1.2f" % (nprot_SC*1./N)


print
print "The total fraction bound is %1.2f" % ((nprot_HC+nprot_SC)*1./N)
