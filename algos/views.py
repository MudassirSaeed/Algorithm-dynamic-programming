import csv
import random
import sys

from django.shortcuts import render

# Create your views here.
def get_main_page(request):
    return render(request, "main.html")

#Coin-Change-------------start
def coin_change(request):
    i = 0
    f = open("static/inputs/Coin-Change/" + request.POST["file_coin_change"] + ".txt")
    coins = f.readline()
    coins = coins.split()
    for x in coins:
        coins[i] = int(x)
        i += 1
    m = len(coins)
    r_num = random.randrange(0, 2)
    V = 0
    if r_num == 0:
        V = 264
    else:
        V = 350
    # table[i] will be storing the minimum
    # number of coins required for i value.
    # So table[V] will have result
    table = [0 for i in range(V + 1)]

    # Base case (If given value V is 0)
    table[0] = 0

    # Initialize all table values as Infinite
    for i in range(1, V + 1):
        table[i] = sys.maxsize

    # Compute minimum coins required
    # for all values from 1 to V
    for i in range(1, V + 1):

        # Go through all coins smaller than i
        for j in range(m):
            if (coins[j] <= i):
                sub_res = table[i - coins[j]]
                if (sub_res != sys.maxsize and
                        sub_res + 1 < table[i]):
                    table[i] = sub_res + 1
    context = {
        "ans_coin_change": table[V]
    }
    return render(request, "main.html", context)
#Coin-Change-------------end

#longestCommonSubsequence-------------start
def lcs(request):
    i = 0
    st = ["a", "b"]
    f = open("static/inputs/LCS/"+request.POST["file_lcs"]+".txt")
    for x in f:
        st[i] = x
        i = i + 1

    X = st[0]
    Y = st[1]

    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

                # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    context = {
        "ans_lcs":L[m][n]
    }
    return render(request,"main.html",context)
#longestCommonSubsequence-------------end

#matrixChain multiplication-------------start
def mcm(request):
    i = 0
    count = 0
    f = open("static/inputs/MCM/"+request.POST["file_mcm"]+".txt")
    for x in f:
        count = count + 1
    f.close()

    p = []
    f = open("static/inputs/MCM/"+request.POST["file_mcm"]+".txt")
    data = f.readline()
    data = data.split()
    for x in data:
        p.append(int(x))
    n = len(p)
    # For simplicity of the program, one extra row and one
    # extra column are allocated in m[][].  0th row and 0th
    # column of m[][] are not used
    m = [[0 for x in range(n)] for x in range(n)]

    # m[i, j] = Minimum number of scalar multiplications needed
    # to compute the matrix A[i]A[i + 1]...A[j] = A[i..j] where
    # dimension of A[i] is p[i-1] x p[i]

    # cost is zero when multiplying one matrix.
    for i in range(1, n):
        m[i][i] = 0

    # L is chain length.
    for L in range(2, n):
        for i in range(1, n - L + 1):
            j = i + L - 1
            m[i][j] = sys.maxsize
            for k in range(i, j):

                # q = cost / scalar multiplications
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
    context = {
        "ans_mcm": m[1][n-1]
    }
    return render(request, "main.html", context)
#matrixChain multiplication-------------end

#WORD BREAK#start
def wordBreak(dict, str):
    # return true if we have reached the end of the String,
    if not str:
        return True

    for i in range(1, len(str) + 1):

        # consider all prefixes of current String
        prefix = str[:i]

        # return true if prefix is present in the dictionary and remaining
        # also forms space-separated sequence of one or more
        # dictionary words
        if prefix in dict and wordBreak(dict, str[i:]):
            return True

    # return false if the can't be segmented
    return False


def findWord(request):
    f = open("static/inputs/Word-Break/" + request.POST["file_word_break"] + ".txt")
    data = f.readline()
    arr = data.split()
    # input String
    r_num = random.randrange(0, 2)
    str = ""
    if r_num == 0:
        str = "ibrahim"
    else:
        str = "mudassir"
    if wordBreak(arr, str):
        context = {
            "ans_word_break": "String can be segmented"
        }
    else:
        context = {
            "ans_word_break": "String can't be segmented"
        }
    return render(request, "main.html", context)
#WORD BREAK#End

#Knapsack----------start
def knapsack(request):
    i = 0
    f = open("static/inputs/Knapsack/" + request.POST["file_knapsack"] + ".txt")
    line = f.readline()
    wt = line.split()
    line = f.readline()
    val = line.split()
    for x in wt:
        wt[i] = int(x)
        i+=1
    i=0
    for y in val:
        val[i] = int(y)
        i+=1
    r_num = random.randrange(0, 2)
    W = 0
    if r_num == 0:
        W = 264
    else:
        W = 350
    n = len(val)
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    context = {
        "ans_knapsack": K[n][W]
    }
    return render(request, "main.html", context)
#Knapsack----------end

#longest_increasing_sub-sequence---------------start
def lis(request):
    i = 0
    f = open("static/inputs/LIS/" + request.POST["file_lis"] + ".txt")
    arr = f.readline()
    arr = arr.split()
    for x in arr:
        arr[i] = int(x)
        i = i + 1
    n = len(arr)

    # Declare the list (array) for LIS and initialize LIS
    # values for all indexes
    lis = [1] * n

    # Compute optimized LIS values in bottom up manner
    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j] + 1

    # Initialize maximum to 0 to get the maximum of all
    # LIS
    maximum = 0

    # Pick maximum of all LIS values
    for i in range(n):
        maximum = max(maximum, lis[i])

    context = {
        "ans_lis": maximum
    }
    return render(request, "main.html", context)
#longest_increasing_sub-sequence---------------end

#partition_problem------------start
def findPartition(arr, n):
    sum = 0
    i, j = 0, 0

    # calculate sum of all elements
    for i in range(n):
        sum += arr[i]

    if sum % 2 != 0:
        return False

    part = [[True for i in range(n + 1)]
            for j in range(sum // 2 + 1)]

    # initialize top row as true
    for i in range(0, n + 1):
        part[0][i] = True

    # initialize leftmost column,
    # except part[0][0], as 0
    for i in range(1, sum // 2 + 1):
        part[i][0] = False

    # fill the partition table in
    # bottom up manner
    for i in range(1, sum // 2 + 1):

        for j in range(1, n + 1):
            part[i][j] = part[i][j - 1]

            if i >= arr[j - 1]:
                part[i][j] = (part[i][j] or
                              part[i - arr[j - 1]][j - 1])

    return part[sum // 2][n]


def check_partition(request):
    i = 0
    f = open("static/inputs/Partition-Problem/" + request.POST["file_check_partition"] + ".txt")
    arr = f.readline()
    arr = arr.split()
    for x in arr:
        arr[i] = int(x)
        i = i + 1
    n = len(arr)

    # Function call
    if findPartition(arr, n) == True:
        context = {
            "ans_check_partition": "Can be divided into two subsets of equal sum"
        }
    else:
        context = {
            "ans_check_partition": "Can not be divided into two subsets of equal sum"
        }
    return render(request, "main.html", context)
#partion_problem---------------end

#Rod-Cutting-------------start
INT_MIN = -32767

# Returns the best obtainable price for a rod of length n and
# price[] as prices of different pieces
def rod_cutting(request):
    i = 0
    f = open("static/inputs/Rod-Cutting/" + request.POST["file_rod_cutting"] + ".txt")
    price = f.readline()
    price = price.split()
    for x in price:
        price[i] = int(x)
        i = i + 1
    n = len(price)
    val = [0 for x in range(n + 1)]
    val[0] = 0

    # Build the table val[] in bottom up manner and return
    # the last entry from the table
    for i in range(1, n + 1):
        max_val = INT_MIN
        for j in range(i):
            max_val = max(max_val, price[j] + val[i - j - 1])
        val[i] = max_val

    context = {
        "ans_rod_cutting" : val[n]
    }
    return render(request, "main.html", context)
#Rod-Cutting---------------end

#Shortest Common Supersequence-------------start
# Returns length of LCS for
# X[0..m - 1], Y[0..n - 1]
def lcs_scs(X, Y, m, n):
    L = [[0] * (n + 2) for i in
         range(m + 2)]

    # Following steps build L[m + 1][n + 1]
    # in bottom up fashion. Note that L[i][j]
    # contains length of LCS of X[0..i - 1]
    # and Y[0..j - 1]
    for i in range(m + 1):

        for j in range(n + 1):

            if (i == 0 or j == 0):
                L[i][j] = 0

            elif (X[i - 1] == Y[j - 1]):
                L[i][j] = L[i - 1][j - 1] + 1

            else:
                L[i][j] = max(L[i - 1][j],
                              L[i][j - 1])

    # L[m][n] contains length of
    # LCS for X[0..n - 1] and Y[0..m - 1]
    return L[m][n]

def scs(request):
    i = 0
    st = ["a", "b"]
    f = open("static/inputs/SCS/"+request.POST["file_scs"]+".txt")
    for x in f:
        st[i] = x
        i = i + 1

    X = st[0]
    Y = st[1]
    m = len(X)
    n = len(Y)
    l = lcs_scs(X, Y, m, n)

    # Result is sum of input string
    # lengths - length of lcs
    context = {
        "ans_scs": m + n - l
    }
    return render(request, "main.html",context)
#Shortest Common Supersequence---------------end

#Edit-Distance-------------start
def edit_distance(request):
    i = 0
    st = ["a", "b"]
    f = open("static/inputs/Edit-Distance/"+request.POST["file_edit_distance"]+".txt")
    for x in f:
        st[i] = x
        i = i + 1

    str1 = st[0]
    str2 = st[1]
    m = len(str1)
    n = len(str2)
    # Create a table to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):

            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace

    context = {
        "ans_edit_distance" : dp[m][n]
    }
    return render(request, "main.html",context)
#Edit-Distance---------------end