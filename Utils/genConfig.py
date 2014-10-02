#!/usr/bin/python

''' Script that generates a cubic box full of spheres of radius R0 in a 
cubic lattice with a centered sphere of radius RNP '''

import math
import numpy as np
import string
import random
import argparse

''' FileName '''
fileName = 'config/init.dat'

''' Fraction of each protein '''
P1 = 0.9
P2 = 1 - P1
P3 = 0

parser = argparse.ArgumentParser()

''' General parameters, temperature in [kb*T] and lengths in [nm] '''
parser.add_argument("-kT",       type=float, help="Temperature",              default=1.00)
parser.add_argument("-N",        type=int,   help="Number of particles",      default=128*100)
parser.add_argument("-L",        type=float, help="Lateral size of the box",  default=400.0) 
parser.add_argument("-RNP",      type=float, help="Radius of the NP",         default=50.) # L > 4*RNP

''' Hard-core radius of the proteins in [nm] '''
parser.add_argument("-R1",       type=float, help="Hard Radius of prot 1",    default=6.58)
parser.add_argument("-R2",       type=float, help="Hard Radius of prot 2",    default=3.60)
parser.add_argument("-R3",       type=float, help="Hard Radius of prot 3",    default=4.00)

''' Soft-core radius of the proteins in [nm] '''
parser.add_argument("-Rs1",      type=float, help="Soft Radius of prot 1",    default=11.)
parser.add_argument("-Rs2",      type=float, help="Soft Radius of prot 2",    default=3.60)
parser.add_argument("-Rs3",      type=float, help="Soft Radius of prot 3",    default=4.00)

''' Masses of the proteins in [kDa] '''
parser.add_argument("-M1",       type=float, help="Mass of prot 1",           default=340.00) 
parser.add_argument("-M2",       type=float, help="Mass of prot 2",           default=67.00)
parser.add_argument("-M3",       type=float, help="Mass of prot 3",           default=90.00)

''' Sruface affinity of each protein in [Kb*T] '''
parser.add_argument("-EpsNP1",   type=float, help="Int. energy NP-prot 1",    default=14.0)  
parser.add_argument("-EpsNP2",   type=float, help="Int. energy NP-prot 2",    default=6.00)
parser.add_argument("-EpsNP3",   type=float, help="Int. energy NP-prot 3",    default=11.00)

''' Protein-protein interaction in the surface in [Kb*T] '''
parser.add_argument("-EpsProt1", type=float, help="Attr. energy prot-prot 1", default=0.00)  #Eps_pp=4kT
parser.add_argument("-EpsProt2", type=float, help="Attr. energy prot-prot 2", default=2.00)
parser.add_argument("-EpsProt3", type=float, help="Attr. energy prot-prot 3", default=4.00)

''' Electrical Double layer interaction in [Kb*T] '''
parser.add_argument("-EpsEDL1", type=float, help="EDL energy prot-NP 1",      default=1.501)
parser.add_argument("-EpsEDL2", type=float, help="EDL energy prot-NP 2",      default=0.864)
parser.add_argument("-EpsEDL3", type=float, help="EDL energy prot-NP 3",      default=4.00)

args, unknown = parser.parse_known_args()

''' Nombre total (final) de molecules '''
N0 = args.N

''' A continuacio es determinen els paramteres de les molecules del sistema,
en forma de llistes, de manera que cada llista te tants membres com classes
de molecules volem al sistema '''

'''  [Fib,      IgG,      Alb]  '''
R  = [args.R1,  args.R2,  args.R3]   # Tamany (hard-core) de les molecules 
Rs = [args.Rs1, args.Rs2, args.Rs3]  # Tamany (soft-core) de les molecules '''
M  = [args.M1,  args.M2,  args.M3]   # Massa de les molecules '''

kT = args.kT # Temperatura del sistema

EPS_NP   = [args.EpsNP1*kT, args.EpsNP2*kT, args.EpsNP3*kT] # Energia d'interaccio amb la NP 
EPS_Prot = [args.EpsProt1*kT, args.EpsProt2*kT, args.EpsProt3*kT] # Energia d'interaccio entre proteines adsorbides 
EPS_EDL  = [args.EpsEDL1*kT, args.EpsEDL2*kT, args.EpsEDL3*kT]

''' Tamany i massa de la nano-particula (parametre unic) '''
R1 = args.RNP
M1 = 1000.

''' Tamany de la caixa cubica '''
L = args.L
V = L*L*L
den0 = 1./V
print("Total volume of the box %1.6lf" % V)

''' Nombre de llocs a la xarxa cubica on colocar les prot. '''
n = (int) (L / (1.5*R[0])) # L'espaiat es tria en termes de la prot. mes grossa 
l = L/n # Tamany de cada caixeta de cada molecula 

def dist(x,y,z,L):
	x -= 0.5
	y -= 0.5
	z -= 0.5
	
	return L * math.sqrt(x*x + y*y + z*z)

nTypes = len(R)

class mol():
	""" class constructor"""
	def __init__(self):
		self.molList = []

molType = []

for i in range(nTypes):
	molType.append(mol())

coordsList = [] # llista amb totes les coordenades que ja hem omplert

''' Nombre final de molecules acceptades '''
N = 0

random.seed()

while 1:
	
	randnum = random.random()
	
	""" Els numeros magics, 0.871 i 0.116 mantenen la correcta proporcio de proteines """
	
	if randnum < P1:
		currentType = 0
	elif randnum < P1 + P2:
		currentType = 1
	else:
		currentType = 2
	
	''' Intentem col.locar la proteina escollida en algun lloc lliure de la caixa  '''
	while 1:
		x = ((int)(random.random() * n) * l + l/2) / L
		y = ((int)(random.random() * n) * l + l/2) / L
		z = ((int)(random.random() * n) * l + l/2) / L
		
		''' Si el punt escollit es troba fora de la NP i no ha estat escollit previament '''
		if (( currentType == 0 and dist(x,y,z,L) > R1 + 2.*R[0]) or (currentType != 0 and dist(x,y,z,L) > (R1 + R[currentType]))) and ([x,y,z] not in coordsList):
			(molType[currentType].molList).append([x,y,z])
			coordsList.append([x,y,z])
			N += 1
			break
	
	if(N == N0):
		break


print("Number of accepted molecules: ",N)

f1 = open(fileName,'w')                     # arxiu de sortida: init.dat
f1.write('%d\t%f\t%f\t%f\n' % (N+1,den0,L,V)) # capc,alera de l'arxiu

print("Volume: %1.3e L" % (V * (1e-8)**3))

V_NP = 4./3*3.1416*math.pow(50e-7, 3) # cm3
M_NP = 2.65 * V_NP                    # g/cm3 * cm3 = grams
conc_NP = M_NP/(V * 1e-24)            # g/L
conc_NP /= 10.                        # g/dL

print("NP concentration %1.3lf g/dl" % conc_NP)

# escriptura a arxiu amb les caracteristiques i coordenades de cada proteina
for i in range(nTypes):
	
	N = len(molType[i].molList)
	
	conc = N * M[i] * 1e3 / 6.023e23 # grams
	conc /= V * (1e-8)**3            # g/L
	conc /= 10                       # g/dL
	
	print("%d\tmolecules of type %d with concentration %1.5lf g/dl" % (N,i,conc))
	
	for coords in molType[i].molList:
		f1.write('%d' % i)
		for coord in coords:
			f1.write('\t%1.8f' % (coord*L))
		f1.write('\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\n' % (R[i],Rs[i],M[i],EPS_NP[i],EPS_EDL[i],EPS_Prot[i]))

# la darrera linia es la corresponent a la NP
f1.write('%d\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f%1.8f\n' % (nTypes, 0.5*L,0.5*L,0.5*L, R1, 0, M1, 0, 0))

f1.close()
