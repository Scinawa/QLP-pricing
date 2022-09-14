import numpy as np
import scipy as sp
from scipy import stats
import pdb
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

def gibbs_distrib(v):
    numerator = np.exp(v)
    return numerator/np.linalg.norm(numerator, 1)

def gibbs_distribution_v2(v):
    max_entry = np.max(v)
    v_ = v - np.ones(len(v))*max_entry
    numerator = np.exp(v_)
    return numerator/np.linalg.norm(numerator, 1)

# why do we need these two functions? just to check 
# (even if we know theoretically that they are equivalent) that they are also numerically equivalent
# np.isclose(zs.gibbs_distrib(F.dot(y)), zs.gibbs_distribution_v2(F.dot(y)))
# True 



def calc_probabilities(A,x,y, sign=True):

    if sign:
        u = np.dot(A.T,x)    
    else: 
        u = np.dot(-A.T,x)    

    u_max = np.max(u)

    v = np.dot(A,y)
    v_max = np.max(v)

    # 
    px = np.sum([np.exp(ui-u_max) for ui in u  ] ) / len(y)    # px it's divided by length y!
    py = np.sum([np.exp(vi-v_max) for vi in v  ] ) / len(x)    # py it's divided by lenght x!

    if px>1 or py>1:
        print("px,py bigger than 1: {} {} ".format(px,py))
    
    return (px, py)


def create_non_arbitrage_prices(pis_fake, n, k, mus, sigmas, omegas): 
    S = build_S(mus, sigmas, omegas)
    attempt = np.random.rand(n)

    x = np.random.rand(k)
    x_ = x / np.linalg.norm(x, 1)

    pis = S.dot(x_)

    return pis


def build_D(omegas, mus, sigmas, pis, choice, K):
    def basket_value(omega):
        basket_value = np.sum( [ pi * np.exp(omega*sigmai + mui) if is_in_basket==1 else 0  for mui,sigmai,pi,is_in_basket in zip(mus,sigmas,pis,choice)    ]  ) 
        return basket_value
    
    D = np.array( [   max(0, basket_value(omega) - K ) for omega in omegas]     )

    return D

def build_S(pis, mus, sigmas, omegas):
    S = np.array(  [ np.array([ pi * np.exp(sigma*omega+ mu) for omega in omegas   ]) for pi, mu,sigma in zip(pis, mus, sigmas) ] ) 
    return S


def build_F(n, k, mu, sigma, omegas, pis, D , alpha=1, R=1):
    """
    🤬🤬🤬 TODO BE SURE WE ARE NORMALIZING EVERYTHING 🤬🤬🤬
    """
    # idea: we stack first all the rows except the last column
    # and then we hstack the last column. 

    S = build_S(pis, mu, sigma, omegas)     # Shuould we return smax for scaling? 
    S_  = S/np.max(S)

    # first two rows of F
    F = np.ones(k+2-1)                          
    F = np.vstack((F, -1*np.ones(k+2-1) ))      
    # note k+2-1 as we add the last column at the end

    # third row of F, with matrix D 
    # note that D is already normalized 
    D_ = D/np.max(D)
    D_concat = np.hstack(( D_, np.array([0]) ) )   # it's alpha/R which is alpha TODO check 0 or 1?
    F = np.vstack((F, D_concat))
    
    # last block of F: matrix A, which is composed of S matrix in text
    A = np.vstack((S_, -S_, np.ones(k), (-1)*np.ones(k) ))  
    #print("this", S.shape, A.shape)

    A_concat = np.concatenate(  ( A, np.array([ np.zeros(A.shape[0])]).T  ), axis =1 )
    F = np.vstack( (F, A_concat) )


    # let's append last column: (1,1, alpha/R, vec(c)/R ).T    
    # which we also need to normalize 
    pis_ = pis/np.max(pis)
    c = np.concatenate((pis_, -pis_, np.array([1]), np.array([-1])))

    last_column = np.array( [np.concatenate (
                ( np.array( [1, 1, alpha/R] ),  -c/R )
                )])

    F = np.hstack ( (F, last_column.T )  )

    assert np.max(F) <= 1
    assert np.min(F) >= -1

    return (F, D_, A, c)


def hack_pis(S, pis):
    
    solution = np.random.rand(S.shape[1])
    solution = solution / np.linalg.norm(solution, 1)

    hacked_pis = S.dot(solution)

    return hacked_pis

def update_F(F, alpha, N1, N2):
    """
    TODO, just updated the position of the matrix where there is an \alpha. 
    """
    return F




def choice_of_assets(n):
    """return a portfolio of only 1/6 assets """
    max_assets = int(n**1/2)
    choices = np.zeros(n)
    selected = random.sample(range(n), max_assets)
    for i in selected:
        choices[i] = 1

    # super hack to be sure that we have a nonzero element 
    # in our values of the random variable D
    if len(selected)==0:
        choices[0]=1
    
    return choices



####### Auxiliary functions

def rando(rip, n):
    maximum =0 
    for i in range(rip):
        x=np.random.rand(n)
        x=x/np.linalg.norm(x)
        A = np.random.rand(n,n)*2-np.ones(n**2).reshape(n,n)
        max_prob = prob(A,x)
        print("r", max_prob)
        maximum = max(max_prob[0], maximum)
    return maximum




def picco(rip, n):
    maximum =0
    for i in range(rip): 
        x = np.zeros(n)
        x[4] = 1
        A = np.random.rand(n,n)*2-np.ones(n**2).reshape(n,n)
        max_prob =  prob(A, x ) 
        print("p", max_prob)
        maximum = max(max_prob[0], maximum) 
    return maximum



def solve_zsg(F, epsilon, round_numbers=None):
    """
    [summary]
    """
    eta = epsilon/4

    x, y = np.zeros(F.shape[0]), np.zeros(F.shape[1])
    if round_numbers:
        rounds = range(0,int(round_numbers))
    else:
        rounds = range(0, int(4*np.log(F.shape[0]*F.shape[1])/epsilon**2))
    for game_round in tqdm(rounds):
    #for game_round in range(0, int(4*np.log(F.shape[0]*F.shape[1])/epsilon**2)):

        P = np.exp(- F.T.dot(x))
        Q = np.exp(F.dot(y))

        p = P / np.linalg.norm(P, 1)
        q = Q / np.linalg.norm(Q, 1)

        #print(x.shape, p.shape)
        try:
            dist1 = stats.rv_discrete(name="x", values=(np.arange(len(x)), q))
            dist2 = stats.rv_discrete(name="y", values=(np.arange(len(y)), p))
        except Exception as e:
            print("Exception creating probability distribution for sampling")
            pdb.set_trace()

        a, b = dist1.rvs(size=1)[0], dist2.rvs(size=1)[0]

        x[a] = x[a] + eta
        y[b] = y[b] + eta

    value = x.T.dot(F).dot(y)

    return value, x/np.linalg.norm(x, 1), y/np.linalg.norm(y, 1)


def lpsolver_zerosumgames(A, b, c, epsilon, R, r):
    """

    Args:
        A ([type]): [description]
        b ([type]): [description]
        c ([type]): [description]
        epsilon ([type]): [description]
        R ([type]): [description]
        r ([type]): [description]
    """
    # Epsilon_3 from Lemma 12
    epsilon_3 = epsilon/(6*R(r+1))

    # random alpha
    alpha = np.random.rand()*2*R - R

    for binary_search_ in range(int(np.log(R/epsilon))):  # can put 1 for testing
        
        # two extremes of the binary
        l = -R 
        r = R
        
        value, x, y = solve_zsg( update_F(alpha), epsilon_3)

        if value > 0:
            r = alpha
            alpha = (l+alpha)/2
        elif value <= 2*epsilon_3:
            l = alpha
            alpha = (alpha+r)/2

        F = update_F(F, alpha, N1, N2)




























# n, k = 5,7
# A = np.random.rand(n,k)
# A_with_new_row = np.vstack((A, np.ones(k)))
# A_with_new_column = np.c_[  A, np.ones(n) ]
# https://scipython.com/book/chapter-6-numpy/examples/vstack-and-hstack/
