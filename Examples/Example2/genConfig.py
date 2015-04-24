#!/usr/bin/python

''' Script that generates a cubic box full of spheres in a cubic lattice with a 
centered sphere of radius R_NP '''

import math
import numpy as np
import string
import random
import argparse

''' FileName '''
fileName = 'config/init.dat'

parser = argparse.ArgumentParser()

''' General parameters, temperature in [kb*T] and lengths in [nm] '''
parser.add_argument("-kT",       type=float, help="Temperature",              default=1.00)
parser.add_argument("-N",        type=int,   help="Number of particles",      default=256*5)
parser.add_argument("-L",        type=float, help="Lateral size of the box",  default=375.) 
parser.add_argument("-RNP",      type=float, help="Radius of the NP",         default=50.) # L > 4*RNP
parser.add_argument("-C_PBS",    type=float, help="Concentration of PBS in M", default=0.010)

''' Density of the NP [g/cm3] '''
# Silica: 2.65 g/cm3 /// Polystyrene: 1.00 g/cm3
parser.add_argument("-rho_NP", type=float, help="Density of the NP",          default=2.650)

''' Reference starting concentrations [mg/ml] '''
# C_HSAwSiO2(half-coverage) = 0.175
# C_TfwSiO2(half-coverage)  = 0.055
# C_FibwSiO2(half-coverage) = ???
parser.add_argument("-C_NP",    type=float, help="Conc. of NPs",              default=0.001)
parser.add_argument("-C_Prot1", type=float, help="Conc. of Protein 1",        default=1.0)
parser.add_argument("-C_Prot2", type=float, help="Conc. of Protein 2",        default=0.0)
parser.add_argument("-C_Prot3", type=float, help="Conc. of Protein 3",        default=1.0)

''' Number of excess proteins that may be adsorbed to the NP ''' 
parser.add_argument("-Nmin_Prot1", type=float, help="Min Num. Protein 1",     default=100)
parser.add_argument("-Nmin_Prot2", type=float, help="Min Num. Protein 2",     default=0)
parser.add_argument("-Nmin_Prot3", type=float, help="Min Num. Protein 3",     default=400)

''' Hard-core radius of the proteins in [nm] '''
# Rh_HSA = 2.7
# Rh_Tf  = 3.72
# Rh_Fib = 6.6
parser.add_argument("-R1",       type=float, help="Hard Radius of prot 1",    default=6.60) # Fib
parser.add_argument("-R2",       type=float, help="Hard Radius of prot 2",    default=3.72) # Tf
parser.add_argument("-R3",       type=float, help="Hard Radius of prot 3",    default=2.70) # HSA

''' Soft-core radius of the proteins in [nm] '''
# Rs_HSA = 3.6
# Rs_Tf  = 3.72
# Rs_Fib = 8.5
parser.add_argument("-Rs1",      type=float, help="Soft Radius of prot 1",    default=8.50)
parser.add_argument("-Rs2",      type=float, help="Soft Radius of prot 2",    default=3.72)
parser.add_argument("-Rs3",      type=float, help="Soft Radius of prot 3",    default=3.60)

''' Masses of the proteins in [kDa] '''
# M_HSA = 67
# M_Tf  = 80
# M_Fib = 340
parser.add_argument("-M1",       type=float, help="Mass of prot 1",           default=340.00) 
parser.add_argument("-M2",       type=float, help="Mass of prot 2",           default=80.00)
parser.add_argument("-M3",       type=float, help="Mass of prot 3",           default=67.00)

''' Sruface affinity of each protein in [Kb*T] ''' 
# EpsNP_HSAwSiO2 = 9.7
# EpsNP_TfwSiO2  = 8.3
# EpsNP_FibwSiO2 = 8.
parser.add_argument("-EpsNP1", type=float, help="Int. energy NP-prot 1", default=9.0) 
parser.add_argument("-EpsNP2", type=float, help="Int. energy NP-prot 2", default=15.) 
parser.add_argument("-EpsNP3", type=float, help="Int. energy NP-prot 3", default=9.7)

''' Protein-protein interaction in the surface in [Kb*T] '''
# > 0 for multilayer adsorption
parser.add_argument("-EpsProt1", type=float, help="Attr. energy prot-prot 1", default=4.00) 
parser.add_argument("-EpsProt2", type=float, help="Attr. energy prot-prot 2", default=3.75)
parser.add_argument("-EpsProt3", type=float, help="Attr. energy prot-prot 3", default=3.50)

''' Range of the Protein-protein interaction near the surface [nm] '''
parser.add_argument("-K_pp", type=float, help="Prot-prot 3-body range",       default=66.0)

''' Switch ON/OFF the diagonal part of the Prot-Prot interaction energy matrix. 1 = ON, 0 = OFF '''
parser.add_argument("-EpsDiag", type=int, help="Diagonal part of Prot-prot 3-body ", default=1)

''' Electrical Double layer interaction in [Kb*T] '''
parser.add_argument("-EpsEDL1", type=float, help="EDL energy prot-NP 1",      default=0.52)
parser.add_argument("-EpsEDL2", type=float, help="EDL energy prot-NP 2",      default=1.24)
parser.add_argument("-EpsEDL3", type=float, help="EDL energy prot-NP 3",      default=0.69)

''' Z-potential in [meV] '''
parser.add_argument("-ZpotNP", type=float, help="Z-potential of NP",      default=-25.)
parser.add_argument("-Zpot1", type=float, help="Z-potential prot 1",      default=-15.)
parser.add_argument("-Zpot2", type=float, help="Z-potential prot 2",      default=-20.)
parser.add_argument("-Zpot3", type=float, help="Z-potential prot 3",      default=-15.)

args, unknown = parser.parse_known_args()



gamma = [0]*3

# 1 kT = 25.6 meV
gammaNP = math.tanh(args.ZpotNP/(4*args.kT*25.6))
gamma[0] = math.tanh(args.Zpot1/(4*args.kT*25.6))
gamma[1] = math.tanh(args.Zpot2/(4*args.kT*25.6))
gamma[2] = math.tanh(args.Zpot3/(4*args.kT*25.6))

''' Tamany i massa de la nano-particula (parametre unic) '''
R1 = args.RNP

V_NP = 4./3*3.1416*math.pow(R1*1E-7, 3) # cm3
M_NP = args.rho_NP * V_NP                  # g/cm3 * cm3 = grams

''' mass conversion factor '''
kDa_2_mg =  1.6601E-18
mg_2_kDa =  1./1.6601E-18

print "Mass of the NP: %1.1e kDa" % (M_NP*1000 * mg_2_kDa)

C_NP = args.C_NP # mg/ml = g/l

V_box_1NP = 1./(C_NP / M_NP) # l = dm3
V_box_1NP /= pow(10,3)       # m3
V_box_1NP *= pow(1E9,3)      # nm3

L_box_1NP = pow(V_box_1NP, 1./3) # nm

print "Lenght of the experimental box: %1.1f nm" % L_box_1NP


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
#EPS_EDL  = [args.EpsEDL1*kT, args.EpsEDL2*kT, args.EpsEDL3*kT]
EPS_EDL = [0]*3
for i in range(3):
	EPS_EDL[i] = 7.79 * gammaNP * gamma[i] * args.RNP*R[i]/(args.RNP+R[i])

''' Tamany de la caixa cubica '''
L = args.L
V = L*L*L
den0 = 1./V

C_Prot = [args.C_Prot1, args.C_Prot2, args.C_Prot3]   # mg/ml = g/l

''' mass conversion factor '''
kDa_2_mg =  1.6601E-18

M[0] *= kDa_2_mg   # mg
M[1] *= kDa_2_mg   # mg
M[2] *= kDa_2_mg   # mg

''' volume conversion factor '''
ml_2_nm3 = 1E21

N_Ref = [0]*3

''' Number of proteins of each type in reference box '''
N_Ref[0] = int(C_Prot[0] / M[0] / ml_2_nm3 * V_box_1NP )
N_Ref[1] = int(C_Prot[1] / M[1] / ml_2_nm3 * V_box_1NP )
N_Ref[2] = int(C_Prot[2] / M[2] / ml_2_nm3 * V_box_1NP )

print "[Exp] Number of proteins/NP: "
print N_Ref

N_min = [0]*3

N_min[0] = int(N_Ref[0] * V / V_box_1NP)
N_min[1] = int(N_Ref[1] * V / V_box_1NP)
N_min[2] = int(N_Ref[2] * V / V_box_1NP)

print "[Sim] Minimum number of 'free' proteins in simulation box: "
print N_min


normFactor = 0.
if N_min[0] > 0:
	normFactor += N_min[0] + args.Nmin_Prot1
if N_min[1] > 0:
	normFactor += N_min[1] + args.Nmin_Prot2
if N_min[2] > 0:
	normFactor += N_min[2] + args.Nmin_Prot3


''' Fraction of each protein From less to most abundant '''
if N_min[0] > 0:
	P1 = (N_min[0] + args.Nmin_Prot1)*1./normFactor
else:
	P1 = 0.
if N_min[1] > 0:
	P2 = (N_min[1] + args.Nmin_Prot2)*1./normFactor
else:
	P2 = 0.
P3 = 1 - (P1 + P2)

print "[Sim] Proportions of each protein:\n %1.2f %1.2f %1.2f" % (P1, P2, P3)

''' restore the masses to the original units '''
M  = [args.M1,  args.M2,  args.M3]   # Massa de les molecules '''

Lreaction = L*0.8
Lmin = (L - Lreaction)/2
Lmax = L - Lmin

print "[Sim] Box geometry:"
print "Lbox ", L, "nm Lreaction ", Lreaction, "nm boundaries ", Lmin, Lmax
#print("Volume: %1.3e L" % (V * (1e-8)**3))

Rmax = 0.
for r in R:
	if r > Rmax:
		Rmax = r

''' Nombre de llocs a la xarxa cubica on colocar les prot. '''
n = (int) (L / (2.0*Rmax)) # L'espaiat es tria en termes de la prot. mes grossa 
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

nReaction = [0]*3
nReactionRef = [0]*3

nReactionRef[0] = N_min[0]*pow(0.8,3)
nReactionRef[1] = N_min[1]*pow(0.8,3)
nReactionRef[2] = N_min[2]*pow(0.8,3)


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
		
		if nReaction[currentType] >= nReactionRef[currentType]:
			if x*L > Lmin and x*L < Lmax and y*L > Lmin and y*L < Lmax and z*L > Lmin and z*L < Lmax:
				#print "reaction zone is full of: ", currentType, " coords ", x*L, y*L, z*L
				continue
		else:
			if x*L < Lmin or x*L > Lmax or y*L < Lmin or y*L > Lmax or z*L < Lmin or z*L > Lmax:
				continue
		
		''' Si el punt escollit es troba fora de la NP i no ha estat escollit previament '''
		if (( currentType == 0 and dist(x,y,z,L) > R1 + 2.*R[2]) or (currentType != 0 and dist(x,y,z,L) > (R1 + R[currentType]))) and ([x,y,z] not in coordsList):
			(molType[currentType].molList).append([x,y,z])
			coordsList.append([x,y,z])
			N += 1
			
			if x*L > Lmin and x*L < Lmax and y*L > Lmin and y*L < Lmax and z*L > Lmin and z*L < Lmax:
				nReaction[currentType] += 1
			
			break
	
	if(N == N0):
		break

#print nReaction, nReactionRef

print "[Sim] Number of accepted molecules: ", N

f1 = open(fileName,'w')                     # arxiu de sortida: init.dat
f1.write('%d\t%f\t%f\t%d\t%f\t%f\t%f\t%d\t%d\t%d\t\n' % (N+1, args.C_PBS, args.K_pp, args.EpsDiag, L, V, L_box_1NP, N_Ref[0], N_Ref[1], N_Ref[2])) # capc,alera de l'arxiu


# escriptura a arxiu amb les caracteristiques i coordenades de cada proteina
for i in range(nTypes):
	
	N = len(molType[i].molList)
	
	conc = N * M[i] * 1e3 / 6.023e23 # grams
	conc /= V * (1e-8)**3            # g/L
#	conc /= 10            # g/dL
	
	print("%d\tmolecules of type %d with concentration %1.5lf mg/ml" % (N,i,conc))

	conc = N_Ref[i]/V_box_1NP           # nm-3
	
	print("%d\tmolecules of type %d with concentration %1.5e nm-3" % (N,i,conc))
	print("%d\tmolecules of type %d with concentration %1.5e M=1mol/L\n" % (N,i,conc/0.6))

	for coords in molType[i].molList:
		f1.write('%d' % i)
		for coord in coords:
			f1.write('\t%1.8f' % (coord*L))
		f1.write('\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\n' % (R[i],Rs[i],M[i],EPS_NP[i],EPS_EDL[i],EPS_Prot[i]))

# la darrera linia es la corresponent a la NP
f1.write('%d\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f\t%1.8f%1.8f\n' % (nTypes, 0.5*L,0.5*L,0.5*L, R1, 0, 1000, 0, 0))

f1.close()
