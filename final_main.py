import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import genfromtxt
from numpy.linalg import matrix_rank
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import os


############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    # pure colors...
    GREY = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    # color styles...
    HEADER = '\033[95m\033[1m'
    MSG = '\033[95m'
    QUESTION = '\033[93m\033[3m'
    COMMENT = '\033[96m'
    IMPLEMENTED = '\033[92m' + '[IMPLEMENTED] ' + '\033[96m'
    TODO = '\033[94m' + '[TO DO] ' + '\033[96m'
    WARNING = '\033[91m'
    ERROR = '\033[91m\033[1m'
    ENDC = '\033[0m'  # RECOVERS DEFAULT TEXT COLOR
    BOLD = '\033[1m'
    ITALICS = '\033[3m'
    UNDERLINE = '\033[4m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def screen_clear():
    # for mac and linux(here, os.name is 'posix')
    if os.name == 'posix':
        _ = os.system('clear')
    else:
        # for windows platfrom
        _ = os.system('cls')


### A. CONSTRUCTION OF RANDOM GAMES TO SOLVE ###

def generate_random_binary_array(N, K):
    print(bcolors.IMPLEMENTED + '''
    # ROUTINE:  generate_random_binary_array
    # PRE:      N = length of the 1xN binary array to be constructed
    #           K = number of ones within the 1xN binary array
    # POST:     A randomly constructed numpy array with K 1s and (N-K) zeros''' + bcolors.ENDC)

    if K <= 0:  # construct an ALL-ZEROS array
        randomBinaryArray = np.zeros(N)

    elif K >= N:  # construct an ALL-ONES array
        randomBinaryArray = np.ones(N)

    else:
        randomBinaryArray = np.array([1] * K + [0] * (N - K))
        np.random.shuffle(randomBinaryArray)

    return (randomBinaryArray)


def generate_winlose_game_without_pne(m, n, G01, G10, earliestColFor01, earliestRowFor10):
    print(bcolors.IMPLEMENTED + '''
    # ROUTINE:   generate_random_binary_array
    # PRE:       (m,n) = the dimensions of the game to construct
    #            (G10,G01) = numbers of (1,0)-elements and (0,1) elements in the game
    # POST:      Construct a mxn win-lose game randomly, so that:
    #             * There are G10 (1,0)-elements and G01 (0,1)-elements.
    #             * (1,1)-elements are forbidden
    #             * Each row possesses at least one (0,1)-element
    #             * Each column possesses at least one (1,0)-element
    #             * (0,1)-elements lie in columns from earliestColFor01 to n
    #             * 10-elements lie in rows from earliestRowFor10 to n
    # ERROR HANDLING:
    #   [EXITCODE =  0] SUCCESSFUL CREATION OF RANDOM WIN-LOSE GAME 
    #   [EXITCODE = -1] WRONG PARAMETERS 
    #   [EXITCODE = -2] INSUFFICIENT 10-ELEMENTS OR 01-ELEMENTS 
    #   [EXITCODE = -3] TOO MANY 10-ELEMENTS OR 01-ELEMENTS 
    #   [EXITCODE = -4] NOT ENOUGH SPACE TO POSITION 10-ELEMENTS, GIVEN POSITIONS OF 01-ELEMENTS 
    #   [EXITCODE = -5] BAD LUCK, SOME COLUMN WITHIN 10-ELIGIBLE AREA IS ALREADY FILLED WITH 01-ELEMENTS''' + bcolors.ENDC)

    isIntegerFlag = True

    try:
        # try converting to integer
        int(m)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(n)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(G01)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(G10)
    except ValueError:
        isIntegerFlag = False

    try:
        # try converting to integer
        int(earliestColFor01)
    except ValueError:
        isIntegerFlag = False
    try:
        # try converting to integer
        int(earliestRowFor10)
    except ValueError:
        isIntegerFlag = False

    if not isIntegerFlag or np.amin([m, n]) < 2 or np.amax([m, n]) > maxNumberOfActions or m > n or np.amin(
            [earliestRowFor10, earliestColFor01]) < 0 or (earliestRowFor10 > m - 1) or (earliestColFor01 > n - 1):
        # WRONG INPUT PARAMETERS
        print(bcolors.ERROR + "ERROR MESSAGE GEN 1: wrong input parameters" + bcolors.ENDC)
        return (
            -1, np.zeros([maxNumberOfActions, maxNumberOfActions]), np.zeros([maxNumberOfActions, maxNumberOfActions]))

    # initialization of the two payoff matrices...
    R = np.zeros([m, n])
    C = np.zeros([m, n])

    if (G10 < n or G01 < m):
        print(bcolors.ERROR + "ERROR MESSAGE GEN 2: NOT ENOUGH 10-elements and/or 01-elements: G10 =", G10, " < n =", n,
              "? G01 = ", G01, "< m =", m, "?" + bcolors.ENDC)
        return (-2, R, C)

    if G10 > (m - earliestRowFor10) * n or G01 > m * (
            n - earliestColFor01) or G01 + G10 > m * n - earliestRowFor10 * earliestColFor01:
        print(bcolors.ERROR + "ERROR MESSAGE GEN 3: TOO MANY 10-elements and/or 01-elements:" + bcolors.ENDC)
        print("\tG10 =", G10, "> (m-earliestRowFor10)*n =", (m - earliestRowFor10) * n, "?")
        print("\tG01 =", G01, "> m*(n-earliestColFor01) =", m * (n - earliestColFor01), "?")
        print("\tG01+G10 =", G01 + G10, "> m*n - earliestRowFor10*earliestColFor01 =",
              m * n - earliestRowFor10 * earliestColFor01, "?")
        return (-3, R, C)

    # choose the random positions for 01-elements, within the eligible area of the bimatrix...
    # eligible area for 01-elements: rows = 0,...,m-1, columns = earliestColFor01,...,n-1

    # STEP 1: choose m 01-elements, one per row, within the eligible area [0:m]x[earliestColFor01s:n] of the bimatrix.

    numEligibleCellsFor01 = m * (n - earliestColFor01)  # all cells in bimatrix are currently 00-elements

    ArrayForOne01PerRow = np.zeros(numEligibleCellsFor01)
    for i in range(m):
        random_j = np.random.randint(earliestColFor01, n)
        position = (n - earliestColFor01) * i + random_j - (earliestColFor01)
        ArrayForOne01PerRow[position] = 1

    # STEP 2: choose G01 – m 01-elements within the eligible area [0:m]x[earliestColFor01s:n] of the bimatrix
    # differently from those cells chosen in STEP 1.
    binaryArrayFor01s = generate_random_binary_array(numEligibleCellsFor01 - m, G01 - m)

    # Position ALL the 01-elements within the eligible area of the bimatrix...
    for i in range(m):
        for j in range(earliestColFor01, n):
            position = (n - earliestColFor01) * i + j - (earliestColFor01)
            if ArrayForOne01PerRow[position] == 1:
                # insert this enforced 10-element in binArrayFor01s

                if position <= 0:  # checking cell (0,earliestColFor01)...
                    binaryArrayFor01sPrefix = np.array([])
                else:
                    binaryArrayFor01sPrefix = binaryArrayFor01s[0:position]

                if position >= numEligibleCellsFor01:  # checking cell (m,n)...
                    binaryArrayFor01sSuffix = np.array([])
                else:
                    binaryArrayFor01sSuffix = binaryArrayFor01s[position:]

                binaryArrayFor01s = np.concatenate((binaryArrayFor01sPrefix, np.array([1]), binaryArrayFor01sSuffix),
                                                   axis=None)

            # print("next position to check for 01-element:",position,"related to the cell [",i,j,"].")
            if binaryArrayFor01s[position] == 1:
                C[i, j] = 1

    # STEP 3: choose n 10-elements, one per column, within the eligible area [earliestRowFor10s:m]x[0:n] of the bimatrix. They should be different from those cells chosen in STEPS 1+2

    numEligibleCellsFor10 = (m - earliestRowFor10) * n  # all cells in bimatrix are currently 00-elements

    # Count only the (0,0)-elements within eligible area of the bimatrix for 10-elements...
    # eligible area for 10-elements: rows = earliestRowFor10,...,m-1, columns = 0,...,n-1
    numFreeEligibleCellsFor10 = 0

    ArrayForOne10PerCol = np.zeros(numEligibleCellsFor10)

    # Count the non-01-elements within the eligible area of the bimatrix for 10-elements
    for i in range(earliestRowFor10, m):
        for j in range(0, n):
            if C[i, j] == 0:
                numFreeEligibleCellsFor10 += 1

    # print("Actual number for eligible cells for 10-elements: numEligibleCellsFor10 = ",numFreeEligibleCellsFor10)
    if numFreeEligibleCellsFor10 < G10:
        print(
            bcolors.ERROR + "ERROR MESSAGE GEN 4: Not enough space to position all the 10-elements within the selected block of the bimatrix and the random position of the 01-elements" + bcolors.ENDC)
        return (-4, np.zeros([m, n]), np.zeros([m, n]))

    # choose the n random positions of 10-elements, one per column, in positions which are NOT already
    # 01-elements, within the 10-eligible area of the bimatrix
    for j in range(n):
        if sum(C[earliestRowFor10:, j:j + 1]) == n - earliestRowFor10:
            # the j-th row of the 10-eligible area in the bimatrix is already filled with 01-elements
            print(bcolors.ERROR + "ERROR MESSAGE 5: Bad luck, column", j,
                  "of the bimatrix is already filled with 01-elements." + bcolors.ENDC)
            return (-5, np.zeros([m, n]), np.zeros([m, n]))

        Flag_EmptyCellDiscovered = False
        while not Flag_EmptyCellDiscovered:
            random_i = np.random.randint(earliestRowFor10, m)
            if C[random_i, j] == 0:
                Flag_EmptyCellDiscovered = True
        position = n * (random_i - earliestRowFor10) + j
        ArrayForOne10PerCol[position] = 1

    # choose the remaining G10-n random positions for 10-elements, in positions which are NOT already
    # used by 01-elements or other (the necessary) 10-elements, within the eligible area of the bimatrix
    binaryArrayFor10s = generate_random_binary_array(numFreeEligibleCellsFor10 - n, G10 - n)
    # expand the binaryArrayFor10s to cover the entire eligible area for 10-elements, so that
    # all cells which are already 01-elements get 0-value and all cells with a necessary 10-element
    # get 1-value.

    # print("INITIAL length of binaryArrayFor10s is",len(binaryArrayFor10s))
    for i in range(earliestRowFor10, m):
        for j in range(0, n):
            position = n * (i - earliestRowFor10) + j
            if C[i, j] == 1:
                # A 01-element was discovered. Insert a ZERO in binaryArrayFor10s, at POSITION,
                # on behalf of cell (i,j)...

                # print("01-element discovered at position (",i,",",j,"). Inserting an additional ZERO at position ",position)

                if position <= 0:  # checking cell (earliestRowFor10,0)...
                    binaryArrayFor10sPrefix = np.array([])
                else:
                    binaryArrayFor10sPrefix = binaryArrayFor10s[0:position]

                if position >= len(binaryArrayFor10s):  # checking cell (m,n)...
                    binaryArrayFor10sSuffix = np.array([])
                else:
                    binaryArrayFor10sSuffix = binaryArrayFor10s[position:]

                binaryArrayFor10s = np.concatenate((binaryArrayFor10sPrefix, np.array([0]), binaryArrayFor10sSuffix),
                                                   axis=None)

                # print("binaryArrayFor10s[position] =",binaryArrayFor10s[position])

            elif ArrayForOne10PerCol[position] == 1:
                # A necessary 10-element discovered. Insert a new ONE in binaryArrayFor10s, at POSITION,
                # on behalf of cell (i,j)...
                # print("A necessary 10-element was discovered at position (",i,",",j,"). Inserting an additional ONE at position ",position)

                if position <= 0:  # checking cell (earliestRowFor10,0)...
                    binaryArrayFor10sPrefix = np.array([])
                else:
                    binaryArrayFor10sPrefix = binaryArrayFor10s[0:position]

                if position >= len(binaryArrayFor10s):  # checking cell (m,n)...
                    binaryArrayFor10sSuffix = np.array([])
                else:
                    binaryArrayFor10sSuffix = binaryArrayFor10s[position:]

                binaryArrayFor10s = np.concatenate((binaryArrayFor10sPrefix, np.array([1]), binaryArrayFor10sSuffix),
                                                   axis=None)

                # print("binaryArrayFor10s[position] =",binaryArrayFor10s[position])

    # print("ACTUAL length of binaryArrayFor10s is",len(binaryArrayFor10s))

    # Insert the G10 10-elements in the appropriate positions of the bimatrix...
    for i in range(earliestRowFor10, m):
        for j in range(0, n):
            position = n * (i - earliestRowFor10) + j
            # print("next position to check for 10-element:",position,"related to the cell [",i,j,"], with C-value = ",C[i,j],"and binaryArrayFor10s-value = ",binaryArrayFor10s[position])
            if binaryArrayFor10s[position] == 1:
                R[i, j] = 1

    return (0, R, C)


### B. MANAGEMENT OF BIMATRICES ###

def drawLine(lineLength, lineCharacter):
    LINE = '\t'
    consecutiveLineCharacters = lineCharacter
    for i in range(lineLength):
        consecutiveLineCharacters = consecutiveLineCharacters + lineCharacter
    LINE = '\t' + consecutiveLineCharacters
    return (LINE)


def drawBimatrix(m, n, R, C):
    print(bcolors.IMPLEMENTED + '''
    ROUTINE:    drawBimatrix
    PRE:        Dimensions and payoff matrices of a win-lose bimatrix game
    POST:       The bimatrix game, with RED for 10-elements, GREEN for 01-elements, and BLUE for 11-elements
    ''' + bcolors.ENDC)

    for i in range(m):
        # PRINTING ROW i...
        if i == 0:
            print(EQLINE)
        else:
            print(MINUSLINE)

        printRowString = ''

        for j in range(n):
            # PRINTING CELL (i,j)...
            if R[i, j] == 1:
                if C[i, j] == 1:
                    CellString = bcolors.CYAN + "("
                else:
                    CellString = bcolors.RED + "("
            elif C[i, j] == 1:
                CellString = bcolors.GREEN + "("
            else:
                CellString = "("

            CellString += str(int(R[i, j])) + "," + str(int(C[i, j])) + ")" + bcolors.ENDC
            if printRowString == '':
                printRowString = '\t[ ' + CellString
            else:
                printRowString = printRowString + ' | ' + CellString

        printRowString = printRowString + ' ]'
        print(printRowString)

    print(EQLINE)


### ALGORITHMS FOR SOLVING BIMATRIX GAMES

# ALG0: Solver for ZERO-SUM games...

def checkForPNE(m, n, R, C):
    print(bcolors.TODO + '''
    # ROUTINE: checkForPNE
    # PRE:  Two mxn payoff matrices R,C, with real values (not necessarily in [0,1])
    # METHOD:
    # POST: (0,0), if no pure NE exists for(R,C), or else 
    #       a pair of actions (i,j) that constitute a pure NE.
    #''' + bcolors.ENDC)
    maxR = [0] * n
    maxC = [0] * m
    PBRcol = np.zeros([m, n])
    PBRrow = np.zeros([m, n])

    for j in range(n):
        maxR[j] = 0
        for i in range(m):
            if R[i, j] >= maxR[j]:
                maxR[j] = R[i, j]

    for i in range(m):
        maxC[i] = 0
        for j in range(n):
            if C[i, j] >= maxC[i]:
                maxC[i] = C[i, j]

    for j in range(n):
        for i in range(m):
            if R[i, j] == maxR[j]:
                PBRrow[i, j] = 1
            else:
                PBRrow[i, j] = 0

    for i in range(m):
        for j in range(n):
            if C[i, j] == maxC[i]:
                PBRcol[i, j] = 1
            else:
                PBRcol[i, j] = 0

    for i in range(m):
        for j in range(n):
            if ((PBRrow[i, j] == 1) and (PBRcol[i, j] == 1)):
                return (i, j)

    return (-1, -1)


def solveZeroSumGame(m, n, A):
    print(bcolors.IMPLEMENTED + '''
    # ROUTINE: solveZeroSumGame
    # PRE:  An arbirary payoff matrix A, with real values (not necessarily in [0,1])
    # METHOD:
    #    Construct the LP describing the MAGNASARIAN-STONE formulation for the 0_SUM case: R = A, C = -A 
    #    [0SUMLP]  
    #    minmize          1*r           + 1*c +  np.zeros(m).reshape([1,m]@x + np.zeros(n).reshape([1,n]@y 
    #      s.t.
    #           -np.ones(m)*r + np.zeros(m)*c +            np.zeros([m,m])@x +                          R@y <= np.zeros(m), 
    #           np.zeros(n)*r -  np.ones(n)*c +                          C'x +            np.zeros([n,n])@y <= np.zeros(n),
    #                     0*r             0*c +  np.ones(m).reshape([1,m])@x + np.zeros(n).reshape([1,n])@y = 1,
    #                     0*r             0*c + np.zeros(m).reshape([1,m])@x +  np.ones(n).reshape([1,n])@y = 1,
    #                                                                   np.zeros(m) <= x,              np.zeros(n) <= y
    #
    # vector of unknowns is a (1+1+m+n)x1 array: chi = [ r, c, x^T , y^T ], 
    # where r is ROW's payoff and c is col's payoff, wrt the profile (x,y).
    #''' + bcolors.ENDC)

    c = np.block([np.ones(2), np.zeros(m + n)])

    Coefficients_a = np.block([(-1) * np.ones(m), np.zeros(n), np.array([0, 0])])  # 1x(m+n+2) array...
    Coefficients_b = np.block([np.zeros(m), (-1) * np.ones(n), np.array([0, 0])])  # 1x(m+n+2) array...
    Coefficients_x = (np.block([np.zeros([m, m]), (-1) * A, np.ones(m).reshape([m, 1]),
                                np.zeros(m).reshape([m, 1])])).transpose()  # mx(m+n+2) array...
    Coefficients_y = (np.block([A.transpose(), np.zeros([n, n]), np.zeros(n).reshape([n, 1]),
                                np.ones(n).reshape([n, 1])])).transpose()  # nx(m+n+2) array...

    SIGMA0 = (np.block([Coefficients_a.reshape([m + n + 2, 1]), Coefficients_b.reshape([m + n + 2, 1]), Coefficients_x,
                        Coefficients_y]))

    SIGMA0_ub = SIGMA0[0:m + n, :]
    Constants_vector_ub = np.zeros(m + n)

    SIGMA0_eq = SIGMA0[m + n:m + n + 2, :]
    Constants_vector_eq = np.ones(2)

    # variable bounds
    Var_bounds = [(None, None), (None, None)]
    for i in range(m + n):
        Var_bounds.append((0, None))  # type: ignore

    zero_sum_res = linprog(c,
                           A_ub=SIGMA0_ub,
                           b_ub=Constants_vector_ub,
                           A_eq=SIGMA0_eq,
                           b_eq=Constants_vector_eq,
                           bounds=Var_bounds,
                           method='highs', callback=None, options=None, x0=None)

    chi = zero_sum_res.x

    x = chi[2:m + 2]
    y = chi[m + 2:m + n + 2]

    # print("Success in solving 0SUMLP for (X,-X) ?\t", zero_sum_res.success)
    # print("Message of the solver for 0SUMLP ?\t",zero_sum_res.message)
    # print("0SUmLP's objective value (additive-wsne guarantee) \t=\t",zero_sum_res.fun)
    # print("NE point for (X,-X) is ( x=",x.reshape([1,m])," , y=",y.reshape([1,n])," ).")

    return (x, y)


def deleteStrictlyDominatedStrategies(m, n, R, C):
    print(bcolors.TODO + '''
    ROUTINE: removeStrictlyDominatedStrategies
    PRE:    A win-lose bimatrix game, described by the two payoff matrices, with payoff values in {0,1}.
    POST:   The subgame constructed by having all strictly dominated actions removed.
             * Each (0,*)-ROW in the bimatrix must be removed.
             * Each (*,0)-COLUMN in the bimatrix must be removed.
             ''' + bcolors.ENDC)
    reduced_R = R
    reduced_C = C

    # get the index of (0,*) rows
    rows_index = []
    for r in range(m):
        if np.all(R[r, :] == 0):
            rows_index.append(r)

    # get the index of (*,0) columns
    columns_index = []
    for c in range(n):
        if np.all(C[:, c] == 0):
            columns_index.append(c)

    # delete rows contained in rows_index
    for r in reversed(rows_index):
        reduced_R = np.delete(R, r, axis=0)
        reduced_C = np.delete(C, r, axis=0)

    # delete columns contained in columns_index
    for c in reversed(columns_index):
        reduced_C = np.delete(reduced_C, c, axis=1)
        reduced_R = np.delete(reduced_R, c, axis=1)

    m = len(reduced_R)
    n = len(reduced_R[0])

    return (m, n, reduced_R, reduced_C)


def interpretReducedStrategiesForOriginalGame(reduced_x, reduced_y, reduced_R, reduced_C, R, C):
    print(bcolors.TODO + '''
    ROUTINE:    interpretReducedStrategiesForOriginalGame
    PRE:        A profile of strategies (reduced_x,reduced_y) for the reduced 
                game (reduced_R,reduced_C), without (0,*)-rows or (*,0)-columns.
    POST:       The corresponding profile for the original game (R,C).
    ''' + bcolors.ENDC)

    x = reduced_x
    y = reduced_y

    return (x, y)


def generate_randomnumbers(m):
    numbers = []
    for i in range(m - 1):
        numbers.append(random.random())

    numbers.sort()
    numbers.insert(0, 0)
    numbers.append(1)

    random_numbers = []
    for i in range(1, len(numbers)):
        random_numbers.append(numbers[i] - numbers[i - 1])

    return random_numbers


def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transposed_matrix = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    return transposed_matrix


def computeApproximationGuarantees(m, n, R, C, x, y):
    print(bcolors.TODO + '''
    ROUTINE: computeApproximationGuarantees
    PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
            A profile (x,y) of strategies for the two players.
    POST:   The two NASH approximation guarantees, epsAPPROX and epsWSNE in [0,1].''' + bcolors.ENDC)

    epsAPPROX = 0
    epsWSNE = 0
    x_trans = np.array(x).reshape(1, len(x))
    C_trans = transpose_matrix(C)
    maxRy = max(np.dot(R, y))
    xtRy = np.dot(x_trans, np.dot(R, y))
    maxCtx = max(np.dot(C_trans, x))
    Cy = np.dot(C, y)
    xtCy = np.dot(x_trans, Cy)

    support_x = np.nonzero(x)[0]
    support_y = np.nonzero(y)[0]
    while ((maxRy - epsAPPROX > xtRy) or (maxCtx - epsAPPROX > xtCy)):
        epsAPPROX = epsAPPROX + 0.01

    max_Ry = np.max(R @ y)
    min_Ry = np.min(R[support_x] @ y)
    max_CTx = np.max(C.T @ x)
    min_CTx = np.min(C.T[support_y] @ x)
    while ((max_Ry - epsWSNE > min_Ry) or (max_CTx - epsWSNE > min_CTx)):
        epsWSNE = epsWSNE + 0.01

    #while ((max_CTx - epsWSNE > min_CTx)):
     #   epsWSNE = epsWSNE + 0.01

    return (epsAPPROX, epsWSNE)


def approxNEConstructionDMP(m, n, R, C):
    print(bcolors.TODO + '''
    ROUTINE: approxNEConstructionDMP
    PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    POST:   A profile of strategies (x,y) produced by the DMP algorithm.''' + bcolors.ENDC)
    effC = [0] * n
    y = [0] * n

    x_trans = generate_randomnumbers(m)
    x = np.array(x_trans).reshape((len(x_trans), 1))
    for i in range(m):
        for j in range(n):
            if (C[i, j] > 0):
                effC[j] = effC[j] + x_trans[i]
    column_index = effC.index(max(effC))
    y[column_index] = 1
    max_index = np.argmax(R[:, column_index])
    for i in range(m):
        if (i == max_index):
            x[i] = x[i] / 2 + 0.5
        else:
            x[i] = x[i] / 2

    (epsAPPROX, epsWSNE) = computeApproximationGuarantees(m, n, R, C, x, y)

    return (x, y, epsAPPROX, epsWSNE)

def approxNEConstructionFP(m, n, R, C):
    print(bcolors.TODO + '''
    ROUTINE: approxNEConstructionFP
    PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    POST:   A profile of strategies (x,y) produced by the FICTITIOUS PLAY algorithm.''' + bcolors.ENDC)

    row_beliefs = [0] * m
    column_beliefs = [0] * n
    row_beliefs[0] = 1
    column_beliefs[0] = 1
    x_plays = row_beliefs
    y_plays = column_beliefs

    for i in range(1, 100):
        row_best_response = np.argmax(np.dot(y_plays, R.T))
        col_best_response = np.argmax(np.dot(x_plays, C))
        row_beliefs[row_best_response] += 1
        column_beliefs[col_best_response] += 1
        x_plays = [x/i for x in row_beliefs]
        y_plays = [x/i for x in column_beliefs]

    uniform_row_beliefs = np.ones(m) / m
    uniform_column_beliefs = np.ones(n) / n
    x_plays_uniform = uniform_row_beliefs
    y_plays_uniform = uniform_column_beliefs

    for i in range(1, 100):
        uniform_row_best_response = np.max(np.dot(y_plays_uniform, R.T))
        uniform_col_best_response = np.max(np.dot(x_plays_uniform, C))
        row_max_indexes = np.where(np.dot(y_plays_uniform, R.T) == uniform_row_best_response)[0]
        col_max_indexes = np.where(np.dot(x_plays_uniform, C) == uniform_col_best_response)[0]

        for index in row_max_indexes:
            uniform_row_beliefs[index] += 1 / len(row_max_indexes)

        for index in col_max_indexes:
            uniform_column_beliefs[index] += 1 / len(col_max_indexes)

        x_plays_uniform =[x/i for x in uniform_row_beliefs]
        y_plays_uniform = [x/i for x in uniform_column_beliefs]

    x_plays = np.array(x_plays).reshape(len(x_plays), 1)
    x_plays_uniform = np.array(x_plays_uniform).reshape(len(x_plays_uniform), 1)

    epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, x_plays, y_plays)
    uniform_epsAPPROX, uniform_epsWSNE = computeApproximationGuarantees(m, n, R, C, x_plays_uniform, y_plays_uniform)

    return (
        row_beliefs, column_beliefs, uniform_row_beliefs, uniform_column_beliefs, epsAPPROX, epsWSNE, uniform_epsAPPROX,
        uniform_epsWSNE)


def approxNEConstructionDEL(m, n, R, C):
    print(bcolors.TODO + '''
    ROUTINE: approxNEConstructionDEL
    PRE:    A bimatrix game, described by the two payoff matrices, with payoff values in [0,1].
    POST:   A profile of strategies (x,y) produced by the DEL algorithm.''' + bcolors.ENDC)

    x_row, y_row = solveZeroSumGame(m, n, R)  # this example solves (R,-R)
    x_row_trans = np.array(x_row).reshape(1, len(x_row))
    V_row = np.dot(x_row_trans, np.dot(R, y_row))
    y_col, x_col = solveZeroSumGame(n, m, C.T)  # this example solves (C,-C)
    x_col_trans = np.array(x_col).reshape(1, len(x_col))
    V_col = np.dot(x_col_trans, np.dot(C, y_col))

    swap_flag = 0
    if (V_row < V_col):
        tempx_row = x_row
        tempy_row = y_row
        tempR = R
        tempm = m
        x_row = x_col
        y_row = y_col

        x_col = tempx_row
        y_col = tempy_row
        R = C
        C = tempR
        m = n
        n = tempm
        swap_flag = 1

    if V_row <= 2/3:
        if swap_flag==0:
            x_final = x_col
            y_final = y_row
        else:
            x_final = x_row
            y_final = y_col
    elif max((np.dot(np.array(x_row).reshape(1, len(x_row)), C)) <= 2 / 3):
        x_final = x_row
        y_final = y_row
    else:
        j = np.dot(np.array(x_row).reshape(1, len(x_row)), C).index(
            max(np.dot(np.array(x_row).reshape(1, len(x_row)), C)))
        for i in range(m):
            if (R[i, j] > 1 / 3) and (C[i, j] > 1 / 3):
                index = i
        x_final = [0] * m
        y_final = [0] * n
        x_final[index] = 1
        y_final[index] = 1

    if swap_flag:
        tempR = R
        tempm = m
        R = C
        C = tempR
        m = n
        n = tempm

    x_final = np.array(x_final).reshape(len(x_final), 1)

    epsAPPROX, epsWSNE = computeApproximationGuarantees(m, n, R, C, x_final, y_final)

    return (x_final, y_final, epsAPPROX, epsWSNE)


### C. GET INPUT PARAMETERS ###
def determineGameDimensions():
    m = 0
    while m < 2 or m > maxNumberOfActions:
        RowActionsString = input(bcolors.QUESTION + 'Determine the size 2 =< m =< ' + str(
            maxNumberOfActions) + ', for the mxn bimatrix game: ' + bcolors.ENDC)
        if RowActionsString.isdigit():
            m = int(RowActionsString)
            print(bcolors.MSG + "You provided the value m =" + str(m) + bcolors.ENDC)
            if m < 2 or m > maxNumberOfActions:
                print(bcolors.ERROR + 'ERROR INPUT 1: Only positive integers between 2 and ' + str(
                    maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC)
        else:
            m = 0
            print(bcolors.ERROR + 'ERROR INPUT 2: Only integer values between 2 and ' + str(
                maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC)

    n = 0
    while n < 2 or n > maxNumberOfActions:
        ColActionsString = input(bcolors.QUESTION + 'Determine the size 1 =< n =< ' + str(
            maxNumberOfActions) + ', for the mxn bimatrix game: ' + bcolors.ENDC)
        if ColActionsString.isdigit():
            n = int(ColActionsString)
            print(bcolors.MSG + "You provided the value n =" + str(n) + bcolors.ENDC)
            if n < 2 or n > maxNumberOfActions:
                print(bcolors.ERROR + 'ERROR INPUT 3: Only positive integers between 2 and ' + str(
                    maxNumberOfActions) + ' are allowable values for m. Try again...' + bcolors.ENDC)
        else:
            n = 0
            print(bcolors.ERROR + 'ERROR INPUT 4: Only integer values between 2 and ' + str(
                maxNumberOfActions) + ' are allowable values for n. Try again...' + bcolors.ENDC)

    return (m, n)


def determineNumRandomGamesToSolve():
    numOfRandomGamesToSolve = 0
    while numOfRandomGamesToSolve < 1 or numOfRandomGamesToSolve > 10000:
        numOfRandomGamesToSolveString = input(
            bcolors.QUESTION + 'Determine the number of random games to solve: ' + bcolors.ENDC)
        if numOfRandomGamesToSolveString.isdigit():
            numOfRandomGamesToSolve = int(numOfRandomGamesToSolveString)
            print(bcolors.MSG + "You requested to construct and solve " + str(
                numOfRandomGamesToSolve) + " random games to solve." + bcolors.ENDC)
            if n < 2 or m > maxNumberOfActions:
                print(bcolors.ERROR + 'ERROR INPUT 5: Only positive integers between 1 and ' + str(
                    maxNumOfRandomGamesToSolve) + ' are allowable values for m. Try again...' + bcolors.ENDC)
        else:
            numOfRandomGamesToSolve = 0
            print(bcolors.ERROR + 'ERROR INPUT 6: Only integer values between 2 and ' + str(
                maxNumOfRandomGamesToSolve) + ' are allowable values for n. Try again...' + bcolors.ENDC)

    return (numOfRandomGamesToSolve)


def determineNumGoodCellsForPlayers(m, n):
    G10 = 0
    G01 = 0

    while G10 < 1 or G10 > m * n:
        G10String = input(bcolors.QUESTION + 'Determine the number of (1,0)-elements in the bimatrix: ' + bcolors.ENDC)
        if G10String.isdigit():
            G10 = int(G10String)
            print(bcolors.MSG + "You provided the value G10 =" + str(G10) + bcolors.ENDC)
            if G10 < 0 or G10 > m * n:
                print(bcolors.ERROR + 'ERROR INPUT 7: Only non-negative integers up to ' + str(
                    m * n) + ' are allowable values for G10. Try again...' + bcolors.ENDC)
        else:
            G10 = 0
            print(bcolors.ERROR + 'ERROR INPUT 8: Only integer values up to ' + str(
                m * n) + ' are allowable values for G10. Try again...' + bcolors.ENDC)

    while G01 < 1 or G01 > m * n:
        G01String = input(bcolors.QUESTION + 'Determine the number of (0,1)-elements in the bimatrix: ' + bcolors.ENDC)
        if G01String.isdigit():
            G01 = int(G01String)
            print(bcolors.MSG + "You provided the value G01 =" + str(G01) + bcolors.ENDC)
            if G01 < 0 or G01 > m * n:
                print(bcolors.ERROR + 'ERROR INPUT 9: Only non-negative integers up to ' + str(
                    m * n) + ' are allowable values for G01. Try again...' + bcolors.ENDC)
        else:
            G01 = 0
            print(bcolors.ERROR + 'ERROR INPUT 10: Only integer values up to ' + str(
                m * n) + ' are allowable values for G01. Try again...' + bcolors.ENDC)

    return (G10, G01)


### D. PREAMBLE FOR LAB-2 ###

def print_LAB2_preamble():
    screen_clear()

    print(bcolors.HEADER + MINUSLINE + """
                        CEID-NE509 (2022-3) / LAB-2""")
    print(MINUSLINE + """
        STUDENT NAME:           < provide your name here >
        STUDENT AM:             < provide your AM here >
        JOINT WORK WITH:        < provide your partner's name and AM here >""")
    print(MINUSLINE + bcolors.ENDC)

    input("Press ENTER to continue...")
    screen_clear()

    print(bcolors.HEADER + MINUSLINE + """
        LAB-2 OBJECTIVE: EXPERIMENTATION WITH WIN-LOSE BIMATRIX GAMES\n""" + MINUSLINE + """  
        1.      GENERATOR OF INSTANCES: Construct rando win-lose games 
        with given densities for non-(0,0)-elements, and without pure 
        Nash equilibria.                          (PROVIDED IN TEMPLATE)

        2.      BIMATRIX CLEANUP: Remove all STRICTLY DOMINATED actions 
        for the players, ie, all (0,*)-rows, and all (*,0)-columns from 
        the bimatrix.                                (TO BE IMPLEMENTED) 

        3.      Implementation of elementary algorithms for constructing
        strategy profiles that are then tested for their quality as 
        ApproxNE, or WSNE points.                    (TO BE IMPLEMENTED)

        4.      EXPERIMENTAL EVALUATION: Construct P random games, for
        some user-determined input parameter P, and solve each of them 
        with each of the elementary algorithms. Record the observed 
        approximation guarantees (both epsAPPROXNE and epsWSNE) for the 
        provided strategy profiles.                  (TO BE IMPLEMENTED)

        5.      VISUALIZATION OF RESULTS: Show the performances of the 
        algorithms (as approxNE or WSNE constructors), by constructin 
        the appropriate histograms (bucketing the observewd approximation 
        guarantees at one-decimal-point precision).  (TO BE IMPLEMENTED)
    """ + MINUSLINE + bcolors.ENDC)

    input("Press ENTER to continue...")

def chooseExperiment(algorithm):
    screen_clear()

    m, n = determineGameDimensions()

    print("Choose values of G10 and G01 for each experiment:")
    print("Experiment P1: G10 = 20, G01 = 20")
    print("Experiment P2: G10 = 20, G01 = 50")
    print("Experiment P3: G10 = 20, G01 = 70")
    print("Experiment P4: G10 = 35, G01 = 35")
    G10, G01 = determineNumGoodCellsForPlayers(m, n)

    numOfRandomGamesToSolve = determineNumRandomGamesToSolve()

    DMP_Approx_results = []
    DMP_WSNE_results = []
    DMP_results_R = []
    DMP_results_C = []
    DEL_Approx_results = []
    DEL_WSNE_results = []
    DEL_results_R = []
    DEL_results_C = []
    FP_Approx_results = []
    FP_WSNE_results = []
    FP_results_R = []
    FP_results_C = []
    FP_Approx_uniform_results = []
    FP_WSNE_uniform_results = []
    FP_uniform_R = []
    FP_uniform_C = []

    for i in range(numOfRandomGamesToSolve):
        earliestColFor01 = 0
        earliestRowFor10 = 0

        EXITCODE = -5
        numOfAttempts = 0

        # TRY GETTING A NEW RANDOM GAME
        # REPEAT UNTIL EXITCODE = 0, ie, a valid game was constructed.
        # NOTE: EXITCODE in {-1,-2,-3} indicates invalid parameters and exits the program)
        while EXITCODE < 0:
            # EXIT CODE = -4 ==> No problem with parameters, only BAD LUCK, TOO MANY 01-elements within 10-eligible area
            # EXIT CODE = -5 ==> No problem with parameters, only BAD LUCK, ALL-01 column exists within 10-eligible area
            numOfAttempts += 1
            print("Attempt #" + str(numOfAttempts) + " to construct a random game...")
            EXITCODE, R, C = generate_winlose_game_without_pne(m, n, G01, G10, earliestColFor01, earliestRowFor10)

            if EXITCODE in [-1, -2, -3]:
                print(
                    bcolors.ERROR + "ERROR MESSAGE MAIN 1: Invalid parameters were provided for the construction of the random game." + bcolors.ENDC)
                exit()

        # drawBimatrix(m, n, R, C)

        # SEEKING FOR PNE IN THE GAME (R,C)...
        (i, j) = checkForPNE(m, n, R, C)

        if (i, j) != (-1, -1):
            print(bcolors.MSG + "A pure NE (", i, ",", j, ") was discovered for (R,C)." + bcolors.ENDC)
            exit()
        else:
            print(bcolors.MSG + "No pure NE exists for (R,C). Looking for an approximate NE point..." + bcolors.ENDC)

        reduced_m, reduced_n, reduced_R, reduced_C = deleteStrictlyDominatedStrategies(m, n, R, C)

        print(bcolors.MSG + "Reduced bimatrix, after removal of strictly dominated actions:")
        drawBimatrix(reduced_m, reduced_n, reduced_R, reduced_C)

        if algorithm.upper() == 'DMP' or algorithm.upper() == 'ALL':
            ### EXECUTING DMP ALGORITHM...
            x, y, DMPepsAPPROX, DMPepsWSNE = approxNEConstructionDMP(reduced_m, reduced_n, reduced_R, reduced_C)
            DMP_Approx_results.append(round(DMPepsAPPROX, 4))
            DMP_WSNE_results.append(round(DMPepsWSNE, 4))
            DMP_results_R.append(R)
            DMP_results_C.append(C)
            DMPx, DMPy = interpretReducedStrategiesForOriginalGame(x, y, R, C, reduced_R, reduced_C)

        if algorithm.upper() == 'FP' or algorithm.upper() == 'ALL':
            ### EXECUTING FICTITIOUS PLAY ALGORITHM...
            x, y, x_uniform, y_uniform, FPepsAPPROX, FPepsWSNE, uniformFPepsAPPROX, uniformFPepsWNSE = approxNEConstructionFP(
                reduced_m, reduced_n, reduced_R, reduced_C)
            FP_Approx_results.append(round(FPepsAPPROX, 4))
            FP_WSNE_results.append(round(FPepsWSNE, 4))
            FP_results_R.append(R)
            FP_results_C.append(C)
            FP_Approx_uniform_results.append(round(uniformFPepsAPPROX, 4))
            FP_WSNE_uniform_results.append(round(uniformFPepsWNSE, 4))
            FP_uniform_R.append(R)
            FP_uniform_C.append(C)
            FPx, FPy = interpretReducedStrategiesForOriginalGame(x, y, R, C, reduced_R, reduced_C)

        if algorithm.upper() == 'DEL' or algorithm.upper() == 'ALL':
            ### EXECUTING DEL ALGORITHM...
            x, y, DELepsAPPROX, DELepsWSNE = approxNEConstructionDEL(reduced_m, reduced_n, reduced_R, reduced_C)
            if DELepsWSNE >= 0.67:
                drawBimatrix(reduced_m, reduced_n, reduced_R, reduced_C)
                print("x is that: ", x, "\ny is that: ", y)
            DEL_Approx_results.append(round(DELepsAPPROX, 4))
            DEL_WSNE_results.append(round(DELepsWSNE, 4))
            DEL_results_R.append(R)
            DEL_results_C.append(C)
            DELx, DELy = interpretReducedStrategiesForOriginalGame(x, y, R, C, reduced_R, reduced_C)

    if algorithm.upper() == 'DMP' or algorithm.upper() == 'ALL':
        DMP_worst_R = []
        DMP_worst_C = []
        DMP_approx_sorted = sorted(DMP_Approx_results, reverse=True)
        DMP_max_values = DMP_approx_sorted[:2]
        index1 = DMP_Approx_results.index(DMP_max_values[0])
        index2 = DMP_Approx_results.index(DMP_max_values[1])
        DMP_worst_R.append(DMP_results_R[index1])
        DMP_worst_R.append(DMP_results_R[index2])
        DMP_worst_C.append(DMP_results_C[index1])
        DMP_worst_C.append(DMP_results_C[index2])

        DMP_WSNE_worst_R = []
        DMP_WSNE_worst_C = []
        DMP_WSNE_sorted = sorted(DMP_WSNE_results, reverse=True)
        DMP_WSNE_max_values = DMP_WSNE_sorted[:2]
        index1 = DMP_WSNE_results.index(DMP_WSNE_max_values[0])
        index2 = DMP_WSNE_results.index(DMP_WSNE_max_values[1])
        DMP_WSNE_worst_R.append(DMP_results_R[index1])
        DMP_WSNE_worst_R.append(DMP_results_R[index2])
        DMP_WSNE_worst_C.append(DMP_results_C[index1])
        DMP_WSNE_worst_C.append(DMP_results_C[index2])

    if algorithm.upper() == 'FP' or algorithm.upper() == 'ALL':
        FP_worst_R = []
        FP_worst_C = []
        FP_approx_sorted = sorted(FP_Approx_results, reverse=True)
        FP_max_values = FP_approx_sorted[:2]
        index1 = FP_Approx_results.index(FP_max_values[0])
        index2 = FP_Approx_results.index(FP_max_values[1])
        FP_worst_R.append(FP_results_R[index1])
        FP_worst_R.append(FP_results_R[index2])
        FP_worst_C.append(FP_results_C[index1])
        FP_worst_C.append(FP_results_C[index2])

        FP_WSNE_worst_R = []
        FP_WSNE_worst_C = []
        FP_WSNE_sorted = sorted(FP_WSNE_results, reverse=True)
        FP_WSNE_max_values = FP_WSNE_sorted[:2]
        index1 = FP_WSNE_results.index(FP_WSNE_max_values[0])
        index2 = FP_WSNE_results.index(FP_WSNE_max_values[1])
        FP_WSNE_worst_R.append(FP_results_R[index1])
        FP_WSNE_worst_R.append(FP_results_R[index2])
        FP_WSNE_worst_C.append(FP_results_C[index1])
        FP_WSNE_worst_C.append(FP_results_C[index2])

        FP_uniform_worst_R = []
        FP_uniform_worst_C = []
        FP_uniform_approx_sorted = sorted(FP_Approx_uniform_results, reverse=True)
        FP_uniform_max_values = FP_uniform_approx_sorted[:2]
        index1 = FP_Approx_uniform_results.index(FP_uniform_max_values[0])
        index2 = FP_Approx_uniform_results.index(FP_uniform_max_values[1])
        FP_uniform_worst_R.append(FP_uniform_R[index1])
        FP_uniform_worst_R.append(FP_uniform_R[index2])
        FP_uniform_worst_C.append(FP_uniform_C[index1])
        FP_uniform_worst_C.append(FP_uniform_C[index2])

        FP_uniform_WSNE_worst_R = []
        FP_uniform_WSNE_worst_C = []
        FP_uniform_WSNE_sorted = sorted(FP_WSNE_uniform_results, reverse=True)
        FP_uniform_WSNE_max_values = FP_uniform_WSNE_sorted[:2]
        index1 = FP_WSNE_uniform_results.index(FP_uniform_WSNE_max_values[0])
        index2 = FP_WSNE_uniform_results.index(FP_uniform_WSNE_max_values[1])
        FP_uniform_WSNE_worst_R.append(FP_uniform_R[index1])
        FP_uniform_WSNE_worst_R.append(FP_uniform_R[index2])
        FP_uniform_WSNE_worst_C.append(FP_uniform_C[index1])
        FP_uniform_WSNE_worst_C.append(FP_uniform_C[index2])

    if algorithm.upper() == 'DEL' or algorithm.upper() == 'ALL':
        DEL_worst_R = []
        DEL_worst_C = []
        DEL_approx_sorted = sorted(DEL_Approx_results, reverse=True)
        DEL_max_values = DEL_approx_sorted[:2]
        index1 = DEL_Approx_results.index(DEL_max_values[0])
        index2 = DEL_Approx_results.index(DEL_max_values[1])
        DEL_worst_R.append(DEL_results_R[index1])
        DEL_worst_R.append(DEL_results_R[index2])
        DEL_worst_C.append(DEL_results_C[index1])
        DEL_worst_C.append(DEL_results_C[index2])

        DEL_WSNE_worst_R = []
        DEL_WSNE_worst_C = []
        DEL_WSNE_sorted = sorted(DEL_WSNE_results, reverse=True)
        DEL_WSNE_max_values = DEL_WSNE_sorted[:2]
        index1 = DEL_WSNE_results.index(DEL_WSNE_max_values[0])
        index2 = DEL_WSNE_results.index(DEL_WSNE_max_values[1])
        DEL_WSNE_worst_R.append(DEL_results_R[index1])
        DEL_WSNE_worst_R.append(DEL_results_R[index2])
        DEL_WSNE_worst_C.append(DEL_results_C[index1])
        DEL_WSNE_worst_C.append(DEL_results_C[index2])

    # saving the files
    current_dir = os.getcwd()
    new_folder = "EXPERIMENTS"
    new_folder_path = os.path.join(current_dir, new_folder)
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)

    # make the P1-P4 paths and dirs
    P1_path = os.path.join(new_folder_path, "P1")
    P2_path = os.path.join(new_folder_path, "P2")
    P3_path = os.path.join(new_folder_path, "P3")
    P4_path = os.path.join(new_folder_path, "P4")

    if not os.path.exists(P1_path):
        os.mkdir(P1_path)
    if not os.path.exists(P2_path):
        os.mkdir(P2_path)
    if not os.path.exists(P3_path):
        os.mkdir(P3_path)
    if not os.path.exists(P4_path):
        os.mkdir(P4_path)

    # Define the boundaries for the buckets
    bucket_boundaries = [0.000, 0.101, 0.201, 0.301, 0.401, 0.501, 0.601, 0.701, 0.801, 0.901, 1.001]

    # Compute the histogram
    DMPApproxNEHistogram, DMPApproxbin_edges = np.histogram(DMP_Approx_results, bins=bucket_boundaries)
    DMPWSNENEHistogram, DMPWSNEbin_edges = np.histogram(DMP_WSNE_results, bins=bucket_boundaries)
    FPApproxNEHistogram, FPApproxbin_edges = np.histogram(FP_Approx_results, bins=bucket_boundaries)
    FPWSNENEHistogram, FPWSNEbin_edges = np.histogram(FP_WSNE_results, bins=bucket_boundaries)
    FPUniformApproxNEHistogram, FPUniformApproxbin_edges = np.histogram(FP_Approx_uniform_results,
                                                                        bins=bucket_boundaries)
    FPUniformWSNENEHistogram, FPUniformWSNEbin_edges = np.histogram(FP_WSNE_uniform_results, bins=bucket_boundaries)
    DELApproxNEHistogram, DELApproxbin_edges = np.histogram(DEL_Approx_results, bins=bucket_boundaries)
    DELWSNENEHistogram, DELWSNEbin_edges = np.histogram(DEL_WSNE_results, bins=bucket_boundaries)

    if algorithm.upper() == 'DMP' or algorithm.upper() == 'ALL':
        plt.bar(range(len(DMPApproxNEHistogram)), DMPApproxNEHistogram, align='center')
        plt.xticks(range(len(DMPApproxNEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('DMP ApproxNE')
        file_name = "DMPApproxNEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

        plt.bar(range(len(DMPWSNENEHistogram)), DMPWSNENEHistogram, align='center')
        plt.xticks(range(len(DMPWSNENEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('DMP WNSE NE')
        file_name = "DMPWSNENEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

    if algorithm.upper() == 'FP' or algorithm.upper() == 'ALL':
        plt.bar(range(len(FPApproxNEHistogram)), FPApproxNEHistogram, align='center')
        plt.xticks(range(len(FPApproxNEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('FP ApproxNE')
        file_name = "FPApproxNEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

        plt.bar(range(len(FPWSNENEHistogram)), FPWSNENEHistogram, align='center')
        plt.xticks(range(len(FPWSNENEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('FP WSNE NE')
        file_name = "FPWSNENEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

        plt.bar(range(len(FPUniformApproxNEHistogram)), FPUniformApproxNEHistogram, align='center')
        plt.xticks(range(len(FPUniformApproxNEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('FP uniform ApproxNE')
        file_name = "FPUniformApproxNEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

        plt.bar(range(len(FPUniformWSNENEHistogram)), FPUniformWSNENEHistogram, align='center')
        plt.xticks(range(len(FPUniformWSNENEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('FP uniform WSNE NE')
        file_name = "FPUniformWSNENEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

    if algorithm.upper() == 'DEL' or algorithm.upper() == 'ALL':
        plt.bar(range(len(DELApproxNEHistogram)), DELApproxNEHistogram, align='center')
        plt.xticks(range(len(DELApproxNEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('DEL ApproxNE')
        file_name = "DELApproxNEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

        plt.bar(range(len(DELWSNENEHistogram)), DELWSNENEHistogram, align='center')
        plt.xticks(range(len(DELWSNENEHistogram)),
                   ['0.0,0.1', '0.1,0.2', '0.2,0.3', '0.3,0.4', '0.4,0.5', '0.5,0.6',
                    '0.6,0.7', '0.7,0.8', '0.8,0.9', '0.9,1.0'])
        plt.title('DEL WSNE NE')
        file_name = "DELWSNENEHist.jpg"
        if G10 == 20 and G01 == 20:
            plt.savefig(P1_path + "/" + file_name)
        elif G10 == 20 and G01 == 50:
            plt.savefig(P2_path + "/" + file_name)
        elif G10 == 20 and G01 == 70:
            plt.savefig(P3_path + "/" + file_name)
        elif G10 == 35 and G01 == 35:
            plt.savefig(P4_path + "/" + file_name)
        plt.show()
        plt.close()

    # convert histograms to numpy arrays
    DMPApprox_array = np.array(DMPApproxNEHistogram)
    DMPWSNE_array = np.array(DMPWSNENEHistogram)
    FPApprox_array = np.array(FPApproxNEHistogram)
    FPWSNE_array = np.array(FPWSNENEHistogram)
    FPApproxUniform_array = np.array(FPUniformApproxNEHistogram)
    FPWSNEUniform_array = np.array(FPUniformWSNENEHistogram)
    DELApprox_array = np.array(DELApproxNEHistogram)
    DELWSNE_array = np.array(DELWSNENEHistogram)

    if G10 == 20 and G01 == 20:
        if algorithm.upper() == 'DMP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P1_path, "DMPApprox.out"), DMPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWSNE.out"), DMPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWorstR1.out"), DMP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWorstR2.out"), DMP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWorstC1.out"), DMP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWorstC2.out"), DMP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWSNEWorstR1.out"), DMP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWSNEWorstR2.out"), DMP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWSNEWorstC1.out"), DMP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DMPWSNEWorstC2.out"), DMP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')

        if algorithm.upper() == 'FP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P1_path, "FPApprox.out"), FPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWSNE.out"), FPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformApprox.out"), FPApproxUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWSNE.out"), FPWSNEUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWorstR1.out"), FP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWorstR2.out"), FP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWorstC1.out"), FP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWorstC2.out"), FP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWSNEWorstR1.out"), FP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWSNEWorstR2.out"), FP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWSNEWorstC1.out"), FP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPWSNEWorstC2.out"), FP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWorstR1.out"), FP_uniform_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWorstR2.out"), FP_uniform_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWorstC1.out"), FP_uniform_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWorstC2.out"), FP_uniform_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWSNEWorstR1.out"), FP_uniform_WSNE_worst_R[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWSNEWorstR2.out"), FP_uniform_WSNE_worst_R[1], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWSNEWorstC1.out"), FP_uniform_WSNE_worst_C[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "FPUniformWSNEWorstC2.out"), FP_uniform_WSNE_worst_C[1], delimiter=',',
                       fmt='%1.4e')

        if algorithm.upper() == 'DEL' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P1_path, "DELApprox.out"), DELApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWSNE.out"), DELWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWorstR1.out"), DEL_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWorstR2.out"), DEL_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWorstC1.out"), DEL_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWorstC2.out"), DEL_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWSNEWorstR1.out"), DEL_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWSNEWorstR2.out"), DEL_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWSNEWorstC1.out"), DEL_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P1_path, "DELWSNEWorstC2.out"), DEL_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')
    elif G10 == 20 and G01 == 50:
        if algorithm.upper() == 'DMP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P2_path, "DMPApprox.out"), DMPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWSNE.out"), DMPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWorstR1.out"), DMP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWorstR2.out"), DMP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWorstC1.out"), DMP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWorstC2.out"), DMP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWSNEWorstR1.out"), DMP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWSNEWorstR2.out"), DMP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWSNEWorstC1.out"), DMP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DMPWSNEWorstC2.out"), DMP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')

        if algorithm.upper() == 'FP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P2_path, "FPApprox.out"), FPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWSNE.out"), FPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformApprox.out"), FPApproxUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWSNE.out"), FPWSNEUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWorstR1.out"), FP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWorstR2.out"), FP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWorstC1.out"), FP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWorstC2.out"), FP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWSNEWorstR1.out"), FP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWSNEWorstR2.out"), FP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWSNEWorstC1.out"), FP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPWSNEWorstC2.out"), FP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWorstR1.out"), FP_uniform_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWorstR2.out"), FP_uniform_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWorstC1.out"), FP_uniform_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWorstC2.out"), FP_uniform_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWSNEWorstR1.out"), FP_uniform_WSNE_worst_R[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWSNEWorstR2.out"), FP_uniform_WSNE_worst_R[1], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWSNEWorstC1.out"), FP_uniform_WSNE_worst_C[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "FPUniformWSNEWorstC2.out"), FP_uniform_WSNE_worst_C[1], delimiter=',',
                       fmt='%1.4e')

        if algorithm.upper() == 'DEL' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P2_path, "DELApprox.out"), DELApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWSNE.out"), DELWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWorstR1.out"), DEL_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWorstR2.out"), DEL_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWorstC1.out"), DEL_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWorstC2.out"), DEL_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWSNEWorstR1.out"), DEL_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWSNEWorstR2.out"), DEL_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWSNEWorstC1.out"), DEL_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P2_path, "DELWSNEWorstC2.out"), DEL_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')
    elif G10 == 20 and G01 == 70:
        if algorithm.upper() == 'DMP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P3_path, "DMPApprox.out"), DMPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWSNE.out"), DMPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWorstR1.out"), DMP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWorstR2.out"), DMP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWorstC1.out"), DMP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWorstC2.out"), DMP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWSNEWorstR1.out"), DMP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWSNEWorstR2.out"), DMP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWSNEWorstC1.out"), DMP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DMPWSNEWorstC2.out"), DMP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')

        if algorithm.upper() == 'FP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P3_path, "FPApprox.out"), FPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWSNE.out"), FPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformApprox.out"), FPApproxUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWSNE.out"), FPWSNEUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWorstR1.out"), FP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWorstR2.out"), FP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWorstC1.out"), FP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWorstC2.out"), FP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWSNEWorstR1.out"), FP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWSNEWorstR2.out"), FP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWSNEWorstC1.out"), FP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPWSNEWorstC2.out"), FP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWorstR1.out"), FP_uniform_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWorstR2.out"), FP_uniform_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWorstC1.out"), FP_uniform_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWorstC2.out"), FP_uniform_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWSNEWorstR1.out"), FP_uniform_WSNE_worst_R[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWSNEWorstR2.out"), FP_uniform_WSNE_worst_R[1], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWSNEWorstC1.out"), FP_uniform_WSNE_worst_C[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "FPUniformWSNEWorstC2.out"), FP_uniform_WSNE_worst_C[1], delimiter=',',
                       fmt='%1.4e')

        if algorithm.upper() == 'DEL' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P3_path, "DELApprox.out"), DELApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWSNE.out"), DELWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWorstR1.out"), DEL_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWorstR2.out"), DEL_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWorstC1.out"), DEL_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWorstC2.out"), DEL_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWSNEWorstR1.out"), DEL_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWSNEWorstR2.out"), DEL_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWSNEWorstC1.out"), DEL_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P3_path, "DELWSNEWorstC2.out"), DEL_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')
    elif G10 == 35 and G01 == 35:
        if algorithm.upper() == 'DMP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P4_path, "DMPApprox.out"), DMPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWSNE.out"), DMPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWorstR1.out"), DMP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWorstR2.out"), DMP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWorstC1.out"), DMP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWorstC2.out"), DMP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWSNEWorstR1.out"), DMP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWSNEWorstR2.out"), DMP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWSNEWorstC1.out"), DMP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DMPWSNEWorstC2.out"), DMP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')

        if algorithm.upper() == 'FP' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P4_path, "FPApprox.out"), FPApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWSNE.out"), FPWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformApprox.out"), FPApproxUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWSNE.out"), FPWSNEUniform_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWorstR1.out"), FP_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWorstR2.out"), FP_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWorstC1.out"), FP_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWorstC2.out"), FP_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWSNEWorstR1.out"), FP_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWSNEWorstR2.out"), FP_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWSNEWorstC1.out"), FP_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPWSNEWorstC2.out"), FP_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWorstR1.out"), FP_uniform_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWorstR2.out"), FP_uniform_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWorstC1.out"), FP_uniform_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWorstC2.out"), FP_uniform_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWSNEWorstR1.out"), FP_uniform_WSNE_worst_R[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWSNEWorstR2.out"), FP_uniform_WSNE_worst_R[1], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWSNEWorstC1.out"), FP_uniform_WSNE_worst_C[0], delimiter=',',
                       fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "FPUniformWSNEWorstC2.out"), FP_uniform_WSNE_worst_C[1], delimiter=',',
                       fmt='%1.4e')

        if algorithm.upper() == 'DEL' or algorithm.upper() == 'ALL':
            np.savetxt(os.path.join(P4_path, "DELApprox.out"), DELApprox_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWSNE.out"), DELWSNE_array, delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWorstR1.out"), DEL_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWorstR2.out"), DEL_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWorstC1.out"), DEL_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWorstC2.out"), DEL_worst_C[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWSNEWorstR1.out"), DEL_WSNE_worst_R[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWSNEWorstR2.out"), DEL_WSNE_worst_R[1], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWSNEWorstC1.out"), DEL_WSNE_worst_C[0], delimiter=',', fmt='%1.4e')
            np.savetxt(os.path.join(P4_path, "DELWSNEWorstC2.out"), DEL_WSNE_worst_C[1], delimiter=',', fmt='%1.4e')

### MAIN PROGRAM FOR LAB-2 ###

LINELENGTH = 80
EQLINE = drawLine(LINELENGTH, '=')
MINUSLINE = drawLine(LINELENGTH, '-')
PLUSLINE = drawLine(LINELENGTH, '+')

print_LAB2_preamble()

maxNumOfRandomGamesToSolve = 10000

maxNumberOfActions = 20

choice = input('Do you want to load custom (R, C) matrixes? (y/n)')
if choice == 'y':
    file_name_R = input('Enter the file name for R matrix: ')
    file_name_C = input('Enter the file name for C matrix: ')
    R = np.genfromtxt(os.path.join(os.getcwd(), file_name_R), delimiter=',', skip_header=0)
    C = np.genfromtxt(os.path.join(os.getcwd(), file_name_C), delimiter=',', skip_header=0)
    m = len(R)
    n = len(R[0])
else:
    m, n = determineGameDimensions()
    G10, G01 = determineNumGoodCellsForPlayers(m, n)
    numOfRandomGamesToSolve = 1
    earliestColFor01 = 0
    earliestRowFor10 = 0

    EXITCODE = -5
    numOfAttempts = 0

    # TRY GETTING A NEW RANDOM GAME
    # REPEAT UNTIL EXITCODE = 0, ie, a valid game was constructed.
    # NOTE: EXITCODE in {-1,-2,-3} indicates invalid parameters and exits the program)
    while EXITCODE < 0:
        # EXIT CODE = -4 ==> No problem with parameters, only BAD LUCK, TOO MANY 01-elements within 10-eligible area
        # EXIT CODE = -5 ==> No problem with parameters, only BAD LUCK, ALL-01 column exists within 10-eligible area
        numOfAttempts += 1
        print("Attempt #" + str(numOfAttempts) + " to construct a random game...")
        EXITCODE, R, C = generate_winlose_game_without_pne(m, n, G01, G10, earliestColFor01, earliestRowFor10)

        if EXITCODE in [-1, -2, -3]:
            print(
                bcolors.ERROR + "ERROR MESSAGE MAIN 1: Invalid parameters were provided for the construction of the random game." + bcolors.ENDC)
            exit()

    drawBimatrix(m, n, R, C)

choice = input('Do you want to save (R, C) matrixes? (y/n)')
if choice == 'y':
    file_name_R = input('Enter the file name for R matrix: ')
    file_name_C = input('Enter the file name for C matrix: ')
    np.savetxt(file_name_R, R, delimiter=',', fmt='%1.4e')
    np.savetxt(file_name_C, C, delimiter=',', fmt='%1.4e')

drawBimatrix(m, n, R, C)
choice = input('Do you want to check for PNE? (y/n)')
if choice == 'y':
    i, j = checkForPNE(m, n, R, C)
    if (i, j) != (-1, -1):
        print(bcolors.MSG + "A pure NE (", i, ",", j, ") was discovered for (R,C)." + bcolors.ENDC)
        exit()
    else:
        print(bcolors.MSG + "No pure NE exists for (R,C). Looking for an approximate NE point..." + bcolors.ENDC)

choice = input('Do you want to run an algorithm on (R, C) (y/n):')
if choice == 'y':
    algorithm_name = input("Choose an algorithm to run for the game(DMP, DEL, FP): ")
    if algorithm_name.upper() == 'DMP':
        x, y, epsApprox, epsWSNE = approxNEConstructionDMP(m, n, R, C)
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for DMP:")
        print(MINUSLINE)
        print("\tDMPx =", x, "\n\tDMPy =", y)
        print("\tDMPepsAPPROX =", epsApprox, ".\tDMPepsWSNE =", epsWSNE, "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)
    elif algorithm_name.upper() == 'DEL':
        x, y, epsApprox, epsWSNE = approxNEConstructionDEL(m, n, R, C)
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for DEL:")
        print(MINUSLINE)
        print("\tDELx =", x, "\n\tDELy =", y)
        print("\tDELepsAPPROX =", epsApprox, ".\tDELepsWSNE =", epsWSNE, "." + bcolors.ENDC)
        print(PLUSLINE + bcolors.ENDC)
    elif algorithm_name.upper() == 'FP':
        x, y, x_uniform, y_uniform, epsApprox, epsWSNE, epsApproxUniform, epsWSNEUniform = approxNEConstructionFP(m, n, R, C)
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for FICTITIOUS PLAY:")
        print(MINUSLINE)
        print("\tFPx =", x, "\n\tFPy =", y)
        print("\tFPepsAPPROX =", epsApprox, ".\tFPepsWSNE =", epsWSNE, ".")
        print(PLUSLINE + bcolors.ENDC)
        print(bcolors.MSG + PLUSLINE)
        print("\tConstructed solution for UNIFORM FICTITIOUS PLAY:")
        print(MINUSLINE)
        print("\tFP_Uniform_x =", x_uniform, "\n\tFP_Uniform_y =", y_uniform)
        print("\tFP_Uniform_epsAPPROX =", epsApproxUniform, ".\tFP_Uniform_epsWSNE =", epsWSNEUniform, ".")
        print(PLUSLINE + bcolors.ENDC)

choice = input('Do you want to run a mass experiment(y/n):')
if choice == 'y':
    algorithm = input('Enter the algorithm you want to run:\n1. DMP\n2. DEL\n3. FP\n4. ALL\n->')
    chooseExperiment(algorithm)
