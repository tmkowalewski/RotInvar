# RotInvar
# For calulating rotational invariants from E2 transitions in a level scheme as defined in GOSIA or NuShellX
# Author: Tyler Kowalewski


import numpy as np
import re
from sympy.physics.wigner import wigner_6j as w6J

def formatValue(value, errl, erru):
    # formats value and errors for LaTeX table entry

    # make sure errors are positive
    errl = abs(errl)
    erru = abs(erru)

    if errl == 0 and erru == 0:
        return str(value)
    
    def safe_precision(error):
        if error == 0:
            return 0  # Default precision for 0 error
        else:
            return -int(np.floor(np.log10(error)))

    precision = max(safe_precision(errl), safe_precision(erru))
    
    value = (round(value, precision) if round(value, precision) != 0 else f"{0:.{precision}f}") # get correct precision when reporting 0
    errl = round(errl, precision)
    erru = round(erru, precision)
    if errl == erru:
        return f"{value}({str(errl).replace('.','').replace('0','')[0]})"
    else:
        return f"{value}_{{-{errl:.{precision}f}}}^{{+{erru:0.{precision}f}}}"

def weisskopfBML(ML:str, A:int):
    # estimates Weisskopf unit for a given multipole and mass number
    # returns in si units (e^2fm^2L for E, mu_N^2fm^(2L-2) for M)

    if (ML not in ['M1','E1','E2','E3']):
        raise Exception("Invalid multipole. Must be 'M1', 'E1', 'E2', or 'E3'.")

    L = int(ML[1])

    if (ML[0] == 'M'):
        return 10/np.pi*(1.2)**(2*L-2)*(3/(L+3))**2*A**((2*L-2)/3)
    elif (ML[0] == 'E'):
        return (1.2)**(2*L)/(4*np.pi)*(3/(L+3))**2*A**(2*L/3)

class Level():

    def __init__(self, energy:float, spin:float, parity:int, rindex:int,
                 gindex:int = None, nindex = None, lifetime:float = -1, width:float = 0., 
                 m1moment:float = 0., q_sp:float = 0.) -> None:

        self.energy  = energy       # in MeV
        self.spin = spin            
        self.parity = parity        # 1 for positive, -1 for negative
        self.rindex = rindex        # level notation 2^+_1 <=> spin^parity_rindex
        self.gindex = gindex        # index as assigned by gosia
        self.nindex = nindex        # index as assigned by level energy
        self.lifetime = lifetime    # in ps
        self.width = width          # in eV
        self.m1moment= m1moment     # in u_N
        self.q_sp = q_sp            # in e^2*b
        self.E2diag = (q_sp/(np.sqrt(16*np.pi/5*(self.spin*(2*self.spin-1))/((2*self.spin+1)*(self.spin+1)*(2*self.spin+3)))) if self.spin != 0. else 0.)

        return
    
    def __str__(self) -> str:
        
        return "{0}{1}{2}".format(self.spin,('+' if self.parity == 1 else '-'),self.rindex)
    
    def toLatex(self) -> str:

        spin_str = -1

        if (self.spin.is_integer()):
            spin_str = str(int(self.spin))
        elif ((2*self.spin).is_integer()):
            spin_str = "\\frac{%i}{2}" % (int(self.spin*2))

        return "${0}^{1}_{2}$".format(spin_str,"+" if self.parity == 1 else "-", str(self.rindex))


class Transition():

    def __init__(self, level_f:Level, level_i:Level, elements:dict = None, gamma_en:float = None,
                 branch_ratio:float = None, delta:float = None, gindex:int = None) -> None:
        
        self.level_f = level_f
        self.level_i = level_i
        self.elements = ({"M1":0., "E2":0.} if not elements else elements)  # dict of elements in mu_0, e*b, key corresponds to multipole
        self.gamma_en = gamma_en                                            # in MeV
        self.branch_ratio = branch_ratio                                    # absolute ratio
        self.delta = delta
        self.gindex = gindex                                                # index GOSIA gives to the transition

        return
    
    def getBML(self, ML:str, type:str = 'd', units:str = 'si') -> float:

        if (ML not in ['M1','E1','E2','E3']):
            raise Exception("Invalid multipole. Must be 'M1', 'E1', 'E2', or 'E3'.")
        L = int(ML[1])
        BML = 1/(2*self.level_i.spin + 1)*self.elements[ML]**2

        if (units == 'nat'):
            pass
        elif (units == 'si'):
            if (ML == 'M1'):
                BML *= 1.
            else:
                BML *= 100.**(L)
        elif (units == 'weisskopf'):

            BML /= weisskopfBML(ML, A=70) # A=70 is a placeholder, should be replaced with actual mass number

        else:

            raise Exception("Invalid units. Must be 'nat' (e.g. e^2b^(L/2) and mu^2b^(L-1)), 'si' (e.g. e^2fm^L and mu^2fm^(2L-2)), or 'weisskopf'.")
        
        if type == 'd':
            return BML
        elif type == 'x':
            return BML*(2*self.level_i.spin + 1)/(2*self.level_f.spin + 1)
    
    def calcQsp(self) -> float:

        if (self.level_i == self.level_f):

            return self.elements['E2']/(np.sqrt(16*np.pi/5*(self.level_i.spin*(2*self.level_i.spin-1))/((2*self.level_i.spin+1)*(self.level_i.spin+1)*(2*self.level_i.spin+3))))
        
        else:
            
            print("Error: Only diagonal transitions have a value for Qsp.".format(str(self)))

    def __str__(self) -> str:
        
        return str(self.level_i) + '->' + str(self.level_f)
    
    def toLatex(self) -> str:

        spins = [self.level_i.spin,self.level_f.spin]

        for i in range(len(spins)):
            if spins[i].is_integer():
                spins[i] = int(spins[i])
            elif (spins[i]*2).is_integer():
                spins[i] = "\\frac{%i}{%i}" % (int(spins[i]*2),2)
            else:
                raise Exception("Invalid spin value. Must be integer or half-integer.")
        
        return "${0}^{1}_{2}".format(spins[0],"+" if self.level_i.parity == 1 else "-", self.level_i.rindex) + '\\to' + "{0}^{1}_{2}$".format(spins[1],"+" if self.level_f.parity == 1 else "-", self.level_f.rindex)


class LevelScheme():

    def __init__(self, filename:str = '', filetype:str = '', levels:list = [], transitions:list = []) -> None:

        self.filename = filename
        self.filetype = filetype
        self.levels = levels
        self.transitions = transitions

        self.readFromFile(self.filename, self.filetype)

        self.levels_sorted = sorted(self.levels, key=lambda level: level.energy)
        for i in range(len(self.levels_sorted)):

            self.levels_sorted[i].nindex = i

        return
    
    def getLevelByIpiN(self, spin:float, parity:int, rindex:int) -> Level:
        
        return next((level for level in self.levels if (level.spin == spin and level.parity == parity and level.rindex == rindex)), None)
    
    def getLevelByEnergy(self, energy:float) -> Level:

        return next((level for level in self.levels if (level.energy == energy)), None)
    
    def getLevelByGIndex(self, gindex:int) -> Level:    # gindex is the index assigned by GOSIA

        return next((level for level in self.levels if (level.gindex == gindex)), None)
    
    def getLevelByNIndex(self, nindex:int) -> Level:    # nindex is the index assigned by energy

        return next((level for level in self.levels if (level.nindex == nindex)), None)

    def getTransitionByLevels(self, level_f:Level, level_i:Level) -> Transition:
        
        # transitions are uniquely described by the two levels, so we don't care about the order here
        match = next((transition for transition in self.transitions if (transition.level_f == level_f and transition.level_i == level_i)), None)

        if (match == None):
            match = next((transition for transition in self.transitions if (transition.level_f == level_i and transition.level_i == level_f)), None)

        #if (match == None):

            #print("Warn: getTransitionByLevels didn't find {0}<->{1} in {2}".format(str(level_f),str(level_i),self.filename))
        
        return match

    def getTransitionByGIndex(self, gindex:int) -> Transition:  # gindex is the index assigned by GOSIA

        return next((tran for tran in self.transitions if (tran.gindex == gindex)), None)

    def readFromFile(self, filename:str, filetype:str):

        # reset current scheme
        self.levels = []
        self.transitions = []
        
        if (filetype == "NuShellX"):    # NuShellX decay output filetype: name.deo and name.dei

            # regex pattern for level info, matches
            # float int+/- int float float float float  ----------------------
            level_pattern = re.compile(r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+[0-9]+[\+\-]\s+[0-9]+\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+-+")

            # regex pattern for transition info, matches
            # float int+/- int float float float float float float
            tran_pattern = re.compile(r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+[0-9]+[\+\-]\s+[0-9]+\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?\s+([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?")

            # read in levels and transitions from deo file
            with open(filename + '.deo', "r") as decayfile:

                # deo files do not list ground state, must create it here
                # Energy = 0, Spin  = 0.0, Parity = 1, RelIndex = 1, Lifetime = -1 (Infinite) 
                self.levels.append(Level(0.,0.,1,1,-1.))
                
                for line_num,line in enumerate(decayfile, start = 1):
                    
                    if (level_pattern.search(line)):

                        fields = line.strip().split()

                        energy = float(fields[0])
                        spin = float(fields[1][0])
                        parity = int((1 if fields[1][1] == '+' else -1))
                        rindex = int(fields[2])
                        lifetime = float(fields[3])/(np.log(2))             # listed at T_1/2, need to convert to lifetime
                        width = float(fields[4])
                        m1moment = float(fields[5])
                        q_sp = float(fields[6])/1.E2                        # listed in e^2*fm^2, convert to e^2*b

                        self.levels.append(Level(energy, spin, parity, rindex, lifetime, width, m1moment, q_sp))
                    
                    elif (tran_pattern.search(line)):

                        fields = line.strip().split()

                        energy_f = float(fields[0])
                        spin_f = float(fields[1][:-1])
                        parity_f = int((1 if fields[1][-1] == '+' else -1))
                        rindex_f = float(fields[2])
                        branch_ratio = float(fields[3])
                        gamma_en = float(fields[4])
                        delta = float(fields[5])
                        B_M1 = float(fields[6])
                        B_E2 = float(fields[7])/1.E-4    # given in e^2*fm^4, convert to e^2*b^2

                        level_f = self.getLevelByEnergy(energy_f)
                        level_i = self.levels[-1]
                    
                        transition = Transition(level_f, level_i, gamma_en = gamma_en, branch_ratio = branch_ratio, delta = delta)
                        
                        self.transitions.append(transition)

                        #print("Transition from {0} to {1} added to level scheme.".format(str(level_i),str(level_f)))
                    
                # Substate transitions not listed in deo, so we add them here
                for level in self.levels:

                    self.transitions.append(Transition(level,level,dict(M1=0.,E2=level.E2diag)))

            # Read in matrix elements from dei file
            with open(filename + '.dei', "r") as elementsfile:
                
                for line_num, line in enumerate(elementsfile, start = 1):
                    
                    if ('!' not in line):
                        
                        fields = line.strip().split()

                        value = float(fields[1])/1.E2                   # given in e*fm^2, convert to e*b
                        spin_i = float(fields[2])
                        spin_f = float(fields[4])
                        pole = ('M1' if int(fields[7]) == 1 else 'E2')  # only handles M1 and E2, may need to update
                        rindex_i = int(fields[8])
                        rindex_f = int(fields[9])
                        parity_i = int(fields[10])
                        parity_f = int(fields[11])

                        #skip invalid first line of dei file and go to next line
                        if (parity_i == 0 and parity_f == 0): continue
                        
                        # final, intitial labels opposite dei file b/c we want decay elements, not excitation (may need to check this is working as expected)
                        level_i = self.getLevelByIpiN(spin_i,parity_i,rindex_i)
                        level_f = self.getLevelByIpiN(spin_f,parity_f,rindex_f)

                        try:
                            transition = self.getTransitionByLevels(level_f,level_i)
                            transition.elements[pole] = value
                        except:
                            print("Error: Transition from {0} to {1} not found in level scheme.".format(str(level_i),str(level_f)))
                            continue

        elif (filetype == "GOSIA"):       # GOSIA sum rules filetype: name.smr

            with open(filename,'r') as smrfile:

                # makes the most sense to read line by line, as GOSIA/SIGMA does

                # read in first line which has some info on which elements to use
                level_num, me_num, E2_gindex_i, E2_gindex_f = [int(val) for val in smrfile.readline().strip().split()]

                # smr files do not list relative index, so we need to keep track of it ourselves
                rindex_tracker = dict()
                
                # read in levels
                while (True):

                    fields = smrfile.readline().strip().split()

                    gindex = int(fields[0])
                    spin = float(fields[1])
                    parity = int(1)                  # not given, so set to positive; spin, gindex uniquely defines level
                    energy = float(fields[2])
                    rindex_tracker[spin] = ((rindex_tracker[spin]+1) if (spin in rindex_tracker) else 1)
                    rindex = rindex_tracker[spin]

                    self.levels.append(Level(energy, spin, parity, rindex, gindex))

                    if (gindex == level_num): break

                # read in level coupling scheme (E2 transitions only)
                while (True): 

                    fields = smrfile.readline().strip().split()

                    tran_gindex = int(fields[0])

                    if (E2_gindex_i <= tran_gindex and tran_gindex <= E2_gindex_f): # if this is an E2 element
                        
                        gindex_i = int(fields[1])
                        gindex_f = int(fields[2])

                        levels = [self.getLevelByGIndex(gindex_i),self.getLevelByGIndex(gindex_f)]
                        level_i = levels[0] if levels[0].energy > levels[1].energy else levels[1]   # level with higher energy is initial
                        level_f = levels[1] if levels[0].energy > levels[1].energy else levels[0]   # level with lower energy is final

                        self.transitions.append(Transition(level_f,level_i,gindex = tran_gindex))
                        
                        if (tran_gindex == E2_gindex_f): break
                        

                # read in E2 ME values
                while (True):

                    fields = smrfile.readline().strip().split()

                    tran_gindex = int(fields[0])

                    if (E2_gindex_i <= tran_gindex and tran_gindex <= E2_gindex_f):

                        val = float(fields[1])

                        self.getTransitionByGIndex(tran_gindex).elements['E2'] = val

                        if (tran_gindex == E2_gindex_f): break

        else:

            raise Exception("Invalid filetype. Valid filetypes are 'NuShellX' (name.dei, name.deo) and 'GOSIA' (name.smr).")

        print("Level Scheme read from file! It is shown below.")
        print(str(self))

        return

    def invarByIPiN(self, spin:float, parity:int, rindex:int, invar:str, J:int = 0) -> float:

        # invar values: Q2, Q3, Q4(0), Q4(2), Q4(4), Q5, Q(6)(0), Q6(2), Q6(4), cos3d
        #               Q3cos3d, Q5cos3d(0), Q5cos3d(2), Q5cos3d(4)
        #               cos23d(1), cos23d(2), cos23d(3)

        level = self.getLevelByIpiN(spin, parity, rindex)

        M = np.zeros((len(self.levels_sorted),len(self.levels_sorted))) # E2_matrix[final state nindex][initial state nindex]
        for r in range(len(self.levels_sorted)):
            for v in range(len(self.levels_sorted)):
                transition = self.getTransitionByLevels(self.getLevelByNIndex(r),self.getLevelByNIndex(v))
                if (transition != None):
                    if (r <= v):
                        M[r][v] = transition.elements['E2']
                    else:
                        M[r][v] = (-1)**(self.getLevelByNIndex(r).spin - self.getLevelByNIndex(v).spin)*transition.elements['E2']
                else:
                    M[r][v] = 0.

        ########## INTERMITENT DEFINITIONS ##########

        vL = np.array(M[level.nindex]) # left-hand vector
        vR = np.array(M.T[level.nindex]) # right-hand vector

        def S(J_val) -> np.array:   # (Gosia Manual 9.28c)
            S_mat = np.zeros((len(self.levels_sorted),len(self.levels_sorted)))
            for r in range(len(self.levels_sorted)):
                for v in range(len(self.levels_sorted)):
                    S_mat[r][v] = M[r][v]*w6J(2,2,J_val,level.spin,self.getLevelByNIndex(r).spin,self.getLevelByNIndex(v).spin)
            return S_mat
        
        def ST(J_val) -> np.array:  # (Gosia Manual 9.28d)
            ST_mat = np.zeros((len(self.levels_sorted),len(self.levels_sorted)))
            for r in range(len(self.levels_sorted)):
                for v in range(len(self.levels_sorted)):
                    ST_mat[r][v] = M[v][r]*w6J(2,2,J_val,level.spin,self.getLevelByNIndex(r).spin,self.getLevelByNIndex(v).spin)
            return ST_mat
        
        def T(J_val) -> np.array:   # (Contradicts Gosia Manual 9.28e but gives same output as SIGMA)
            T_mat = np.zeros((len(self.levels_sorted),len(self.levels_sorted)))
            for r in range(len(self.levels_sorted)):
                for v in range(len(self.levels_sorted)):
                    for w in range(len(self.levels_sorted)):
                        T_mat[r][v] += M[r][w]*M[w][v]*w6J(2,2,2,self.getLevelByNIndex(r).spin,self.getLevelByNIndex(v).spin,self.getLevelByNIndex(w).spin)
                    T_mat[r][v] *= w6J(2,2,J_val,level.spin,self.getLevelByNIndex(r).spin,self.getLevelByNIndex(v).spin)
            return T_mat
        
        def Tp() -> np.array:   # (Contradicts Gosia Manual 9.28f but gives same output as SIGMA)
            
            delta = lambda i,j: 1 if i == j else 0

            Tp_mat = np.zeros((len(self.levels_sorted),len(self.levels_sorted)))

            for r in range(len(self.levels_sorted)):
                for v in range(len(self.levels_sorted)):
                    for w in range(len(self.levels_sorted)):
                        Tp_mat[r][v] += M[r][w]*M[w][v]*(-1)**(self.getLevelByNIndex(w).spin - level.spin)*1/(2*self.getLevelByNIndex(v).spin+1)*delta(self.getLevelByNIndex(r).spin,self.getLevelByNIndex(v).spin)
            return Tp_mat

        def P6_1(J_val) -> float:   # (Gosia Manual 9.23)

            return 5*np.sqrt(2*J_val+1)/(2*level.spin+1)*vL @ ST(2).T @ ST(J_val).T @ T(J_val) @ vR

        ########## INAVARIANTS ##########

        def Q2() -> float:  # (Gosia Manual 9.29a)
            
            return float(1/(2*level.spin + 1)*vL @ vL)
        
        def Q3() -> float:  # (Gosia Manual 9.26)

            return float((1/2*(np.sqrt(Q2()) + (Q4(0))**(1/4)))**3)
        
        def Q4(J_val) -> float: # (Contradicts Gosia Manual 9.28c but gives same output as SIGMA)

            def C(J_val) -> float:
                if (J_val == 0):
                    return 5/(2*level.spin + 1)
                elif (J_val == 2 or J == 4):
                    return 35/(2*(2*level.spin + 1))
                else:
                    raise Exception("Invalid J value for C(J) function. Must be 0, 2, or 4.")

            SJ_factor = S(J_val)
            for r in range(len(self.levels_sorted)):
                for v in range(len(self.levels_sorted)):
                    SJ_factor[r][v] *= (-1)**(level.spin + 3*self.getLevelByNIndex(r).spin)

            return float(C(J_val)*vL @ ST(J_val).T @ SJ_factor @ vR)
        
        def Q5(J) -> float: # (Gosia Manual 9.27)

            return float((1/2*((Q4(J))**(1/4) + (Q6(J))**(1/6)))**5)

        def Q6(J_val) -> float: # (Contradicts Gosia Manual 9.29e but agrees with SIGMA)

            def C(J_val) -> float:
                if (J_val == 0):
                    return 5/(2*level.spin + 1)
                elif (J_val == 2 or J == 4):
                    return 35/(2*(2*level.spin + 1))
                else:
                    raise Exception("Invalid J value for C(J) function. Must be 0, 2, or 4.")

            return float(C(J_val)*vL @ ST(J_val).T @ Tp() @ S(J_val) @ vR)
        
        def cos3d() -> float:

            return float(Q3cos3d()/Q3())

        def Q3cos3d() -> float: # (Gosia Manual 9.29b)

            return (-1)**(level.spin.is_integer())*np.sqrt(35/2)*1/(2*level.spin + 1)*vL @ S(2) @ vR
        
        def Q5cos3d(J_val) -> float:    # (Contradicts Gosia Manual 9.29d but agrees with SIGMA)

            def C(J_val) -> float:
                if (J_val == 0):
                    return 5*np.sqrt(35/2)
                elif (J_val == 2 or J == 4):
                    return 35/2*np.sqrt(35/2)
                else:
                    raise Exception("Invalid J value for C(J) function. Must be 0, 2, or 4.")

            return (-1)**(level.spin.is_integer())*C(J)*vL @ ST(J_val).T @ T(J_val) @ vL
        
        def Q6cos23d(J_val) -> float:  # (Contradicts Gosia Manual 9.29g but agrees with SIGMA, J=2,4 nit supported)
                
                if (J_val == 0):
                
                    return float(35/2*P6_1(0))

                else:
                        
                        raise Exception("Invalid J value for Q6cos23d(J) function. Only J=0 is supported")

        if (invar == 'Q2'):

            return Q2()
        
        elif (invar == 'Q3'):

            return Q3()
        
        elif (invar == 'Q4'):

            return Q4(J)
        
        elif (invar == 'Q5'):

            return Q5(J)
        
        elif (invar == 'Q6'):
            
            return Q6(J)
        
        elif (invar == 'cos3d'):

            return cos3d()
        
        elif (invar == 'Q3cos3d'):

            return Q3cos3d()

        elif (invar == 'Q5cos3d'):

            return Q5cos3d(J)
        
        elif (invar == 'Q6cos23d'):

            return Q6cos23d(J)

    def getME(self, spin_f:float, parity_f:int, rindex_f:int, spin_i:float, parity_i:int, rindex_i:int, multipolarity:str = 'E2',) -> float:

        level_f = self.getLevelByIpiN(spin_f, parity_f, rindex_f)
        level_i = self.getLevelByIpiN(spin_i, parity_i, rindex_i)

        return self.getTransitionByLevels(level_f, level_i).elements[multipolarity]

    def strME(self, spin_f:float, parity_f:int, rindex_f:int, spin_i:float, parity_i:int, rindex_i:int, multipolarity:str = 'E2',) -> str:

        level_f = self.getLevelByIpiN(spin_f, parity_f, rindex_f)
        level_i = self.getLevelByIpiN(spin_i, parity_i, rindex_i)

        return "<{0}||{1}||{2}>".format(str(level_f),multipolarity,str(level_i))

    def strLevel(self, spin:float, parity:int, rindex:int) -> str:

        level = self.getLevelByIpiN(spin,parity,rindex)

        if (level == None):

            return 'No such level in the scheme!'

        connected_transitions = [tran for tran in self.transitions if tran.level_i == level or tran.level_f == level]

        string = "Level: {0}\n{1} Transitions:\n".format(str(level),len(connected_transitions))
        
        for i,tran in enumerate(connected_transitions, start = 1):

            string += '{0}: {1}\n'.format(i,tran)

        return string

        return

    def __str__(self) -> str:
        
        string = 'Levels:\n'
        for level in self.levels:

            tran_num = len([tran for tran in self.transitions if tran.level_i == level or tran.level_f == level])

            string += "{0:.3f} MeV\t{1}\t with {2} connecting transitions\n".format(level.energy,str(level),tran_num)

        return string