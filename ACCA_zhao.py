import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import math
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#from scipy.spatial import distance
#from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
#from sklearn import datasets
#iris = datasets.load_iris()
#X = iris['data'][:,:]  
#Y = iris['target']
##Y[Y==2]=1
###fig,ax = plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Set1,
###            edgecolor='k')
##
#fig,ax = plt.subplots();
#ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Set1,
#            edgecolor='k')

#ax.scatter(X2[:,0],X2[:,1],marker='x',c='blue',s=50);

# Read data
file = pd.ExcelFile('attributes.xlsx')
sheet_names = file.sheet_names
df0 = file.parse(sheet_names[5])
df0.set_index('ID',inplace=True)
X1 = np.array(df0)
X = (X1 - X1.mean(axis=0))/ X1.std(axis=0)

##fig = plt.figure()
##ax = fig.add_subplot(111, projection='3d')
##ax.scatter(df0['Rainfall'],df0['BFI'],df0['MAF'])
#mean = (3, 0)
#cov = [[0.3, 1], [1, 0.3]]
#X1 = np.random.multivariate_normal(mean, cov, 50);
#Y1 = 0*np.ones((50,1));
#
#mean = (0, 3)
#cov = [[1, 0.5], [0.5, 1]]
#X2 = np.random.multivariate_normal(mean, cov, 50)
#Y2 = 1*np.ones((50,1));
#
#mean = (1.5, 2.5)
#cov = [[2, 1], [1, 2]]
#X3 = np.random.multivariate_normal(mean, cov, 50)
#Y3 = 2*np.ones((50,1));

#mean = (0.2, 0.1)
#cov = [[0.03, 1], [1, 0.03]]
#X4 = np.random.multivariate_normal(mean, cov, 25);
#Y4 = 4*np.ones((25,1));
#
#mean = (1.2, 0.8)
#cov = [[2, 0.5], [0.5, 2]]
#X5 = np.random.multivariate_normal(mean, cov, 25)
#Y5 = 5*np.ones((25,1));
#
#mean = (0.1, 1.1)
#cov = [[0.2, 0.4], [0.4, 0.2]]
#X6 = np.random.multivariate_normal(mean, cov, 25)
#Y6 = 6*np.ones((25,1));
#
##fig,ax = plt.subplots();
##ax.scatter(X1[:,0],X1[:,1],marker='x',c='green',s=50);
##ax.scatter(X2[:,0],X2[:,1],marker='x',c='blue',s=50);
##ax.scatter(X3[:,0],X3[:,1],marker='x',c='black',s=50);
#
#X = np.concatenate((X1, X2, X3)); #, X4, X5, X6
#Y = np.concatenate((Y1, Y2, Y3)); #, Y4, Y5, Y6
#
#fig,ax = plt.subplots();
#ax.scatter(X[:, 0], X[:, 1], c=Y[:,0], cmap=plt.cm.Set1,
#            edgecolor='k')
#

def euclidean(vector1, vector2):
    '''calculate the euclidean distance, no numpy
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist

def state_transition(tau,M,X,beta,q0):
    # Exploitation
    S1 = np.empty([len(X),1])
    S1[:] = np.nan
    eta = np.empty([X.shape[0],M.shape[0]])
    eta[:] = np.nan
    for c in range(0,M.shape[0]):
        for j in range(0,X.shape[0]):
            try:
                eta[j,c] = 1/euclidean(list(X[j,:]), list(M[c,:]))
            except ZeroDivisionError:
                eta[j,c] = 0
        
    state = tau*(eta**beta)
    ran = np.random.uniform(low=0,high=1,size=(X.shape[0]))
    new = np.argmax(state,axis=1)
    new = np.reshape(new,[len(new),1])
    change = np.argwhere(q0<=ran)
    S1[change] = new[change]
    
    # Biased Exploration
    rem = np.isnan(S1)
    ran2 = np.random.uniform(low=0,high=1,size=(X.shape[0]))
    st_sum = np.sum(state,axis=1)
    p = state/st_sum[:,None]
    cum_p = np.cumsum(p,axis=1)
    select = np.empty([len(X),1])
    select[:] = np.nan
    for i in range(0,len(cum_p)):
        row_p = cum_p[i,:]
        select[i] = int(np.argwhere(row_p>ran2[i]).min())
    S1[rem] = select[rem]
    return S1


def obj_fn(S1,X,N,K):
    try:
        W = np.zeros([N,K])
        for i in range(0,len(W)):
            W[i,np.int0(S1[i,0])] = 1
        mjv = np.empty([K,X.shape[1]])
        mjv[:] = np.nan
    
        for k in range(0,K):
            Choose = S1==k
            X_c = X[Choose[:,0]]
            mjv[k,:] = np.mean(X_c,axis=0)
            
        d = np.empty([X.shape[0],mjv.shape[0]])
        d[:] = np.nan
        for c in range(0,mjv.shape[0]):
            for j in range(0,X.shape[0]):
                d[j,c] = euclidean(list(X[j,:]), list(mjv[c,:]))
        F1 = sum(sum(W*d))
    except TypeError:
        F1 = np.nan
    return F1

def cross_over(L,S_sorted,best,X,N,K):
    L_new = np.array(L)
    S_new = np.array(S)
    fs = np.array(L[0:best])
    ss = S_sorted[:,0:best]
    children = np.empty([ss.shape[0],ss.shape[1]])
    children[:] = np.nan
    f_child = np.empty([ss.shape[1]])
    f_child[:] = np.nan

    for i in range(0,best,2):
        parent1 = ss[:,i]
        parent2 = ss[:,i+1]
        ran3 = np.random.uniform(low=0,high=1,size=(ss.shape[0]))
        change = np.argwhere(ran3>=threshold)
        child1 = np.array(parent1)
        child1[change] = parent2[change]
        child1 = np.reshape(child1,[len(child1),1])
        f_child[i] = obj_fn(child1,X,N,K)
        child2 = np.array(parent2)
        child2[change] = parent1[change]
        child2 = np.reshape(child2,[len(child1),1])
        f_child[i+1] = obj_fn(child2,X,N,K)
        children[:,i] = list(child1)
        children[:,i+1] = list(child2)
    f_child = np.reshape(f_child,[len(fs),1])    
    arg2 = np.argwhere(f_child[:,0]<fs[:,0])
    L_new[arg2] = f_child[arg2]
    S_new[:,arg2] = children[:,arg2]
    return L_new,S_new


def pheromone_update(tau,rho,F_best,S_best):
    try:
        
        W0 = np.zeros([N,K])
        W1 = np.zeros([N,K])
        for i in range(0,len(W0)):
            W0[i,np.int0(S_best[i,0])] = 1
            W1 = W1 + W0*(1/F_best)
        tau_new = tau*(1-rho) + rho*W1
    except TypeError:
        tau_new = tau
        
    return tau_new    


Q0 = np.arange(0.01,0.99,0.08)
T0 = np.arange(0.01,0.99,0.08)
R0 = 0.1
c=0
F_all = np.empty([len(Q0)*len(T0)*1,4])
F_all[:] = np.nan
for x in range(0,len(Q0)):
    for y in range(0,len(T0)):
        for z in range(0,1):


            K = 6
            N = X.shape[0]
            N2 = X.shape[1]
            R = 30
            beta = 2
            q0 = Q0[x]
            threshold = T0[y]
            rho= R0 #[z]
#rho = 0.1
#q0 = 0.8
#threshold = 0.7
            iterations = 50
            
            S_save = np.empty([len(X),iterations])
            S_save[:] = np.nan
            F_save = np.empty([iterations,1])
            F_save[:] = np.nan
            
            
            tau = np.random.uniform(low=0,high=1,size=[N,K])  # Pheromone matrix initialization
            
            M = np.random.uniform(low=0,high=1,size=[K,N2])  # Random Cluster Centers


            for r in range(0,iterations):
                
                F = np.empty([R,1])
                F[:] = np.nan
                S = np.empty([N,R])
                S[:] = np.nan
                for i in range(0,R):
                    S0 = state_transition(tau,M,X,beta,q0)
                    S[:,i] = list(S0)
                    F[i] = obj_fn(S0,X,N,K)
                
                arg = np.argsort(F, axis=0)     
                L = F[arg,0]
                S_sorted = np.array(S[:,arg[:,0]])    
                best = int(0.2*R)    
                
                L_new,S_new = cross_over(L,S_sorted,best,X,N,K)
                arg = np.argsort(L_new, axis=0)   
                F_best = L_new[arg[0]]
                S_best = S_new[:,arg[0]]
                for k in range(0,K):
                    Choose = S_best==k
                    X_c = X[Choose[:,0]]
                    M[k,:] = np.mean(X_c,axis=0)
                tau = pheromone_update(tau,rho,F_best,S_best)
                
                S_save[:,r] = list(S_best)
                F_save[r] = F_best
                print('Iteration No:{} of Set {}'.format(r,c))
            
            F_all[c,0] = F_best.min()
            F_all[c,1] = q0
            F_all[c,2] = threshold
            F_all[c,3] = rho
            c=c+1
            
#            print(c)
"""           
b = np.argwhere(F_save==F_save.min())
Y_hat = S_save[:,b[0,0]]
Y_hat = np.reshape(Y_hat,[len(S_save),1])
#Y = np.reshape(Y,[len(S_save),1])
#for elem in np.concatenate((X, Y, Y_hat),axis=1):
#    if(elem[2]!=elem[3]):
#        ax.scatter(elem[0], elem[1], c='red',s=100);
#
#
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# create polar stereographic Basemap instance.
m = Basemap(projection='merc',llcrnrlat=8,urcrnrlat=25,
            llcrnrlon=72,urcrnrlon=85,
            lat_ts=10,resolution='i')

m.drawcoastlines()
m.drawcountries()
m.scatter(X1[:,1],X1[:,0],latlon=True, c=Y_hat[:,0],cmap=plt.cm.Set1,
       edgecolor='black')
#
#fig2,ax2 = plt.subplots();
#ax2.scatter(X[:,0], X[:,1], c=Y_hat[:,0],cmap=plt.cm.Set1,
#            edgecolor='black')
#
"""
