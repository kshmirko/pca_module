#!/usr/bin/env python

from numpy import array
from numpy import zeros, random

from pca_module import *
import time
import unittest


############## UNIT TESTING ############## 
# Some correct test-values (made 09.05.2007):
# With PCA Module 1.0

# small data set:
X = array([[2, 3, 4, 1],
	   [1, 3, 1, 5],
	   [4, 6, 4, 3],
	   [2, 1, 1, 1],
	   [1, 2, 5, 3],
	   [7, 3, 4, 1]], float)

# mean_center(X) 
X_centered = array([[-0.83333333,  0.,          0.83333333, -1.33333333],
                    [-1.83333333,  0.,        -2.16666667,  2.66666667],
                    [ 1.16666667,  3.,          0.83333333,  0.66666667],
                    [-0.83333333, -2.,         -2.16666667, -1.33333333],
                    [-1.83333333, -1.,          1.83333333,  0.66666667],
                    [4.16666667,  0.,          0.83333333, -1.33333333]])

# standardization(X)                   
X_standardized = array([[0.94573249,  1.96396101,  2.54399491,  0.67082039],
                        [0.47286624,  1.96396101,  0.63599873,  3.35410197],
                        [1.89146497,  3.92792202,  2.54399491,  2.01246118],
                        [0.94573249,  0.65465367,  0.63599873,  0.67082039],
                        [0.47286624,  1.30930734,  3.17999364,  2.01246118],
                        [3.31006371,  1.96396101,  2.54399491,  0.67082039]])




# correct for this X, after calculation (using default parameters): 
# T and P are rotated 180 degrees for NIPALS compared to SVD
Scores_nipals = array([[ 0.40051737, -0.50691688, -0.66225458,  0.61661848],
		       [-2.02073029,  1.12318439,  0.64736695, -0.29402578],
		       [ 1.19358998,  1.70566985,  0.33604821,  0.44214757],
		       [-1.12545468, -1.70021128,  0.56437118,  0.30423703],
		       [-0.34480393,  0.10827787, -1.55227377, -0.44666215],
		       [ 1.89688155, -0.73000396,  0.66674201, -0.62231515]])

Scores_svd = array([[ 0.39124108, -0.51852819, -0.65880069, -0.61661898],
		    [-1.99906634,  1.16627592,  0.63836698,  0.29402557],
		    [ 1.22637448,  1.6839338,   0.32751855, -0.44214676],
		    [-1.1591248,  -1.67453156,  0.57293566, -0.30423691],
		    [-0.34093005,  0.10590213, -1.55329349,  0.44666045],
		    [ 1.88150563, -0.7630521,   0.673273,    0.62231662]])

Loadings_nipals = array([[ 0.6328911,   0.37617472,  0.54163826, -0.40567159],
			 [-0.09435526,  0.69688179,  0.14720258,  0.69554602],
			 [ 0.56210255,  0.22325287, -0.79608775,  0.02105201],
			 [-0.52401018,  0.56833662, -0.22628235, -0.59262392]])

Loadings_svd = array([[ 0.63031124,  0.38949747,  0.54528096, -0.39200527],
		      [-0.10342202,  0.69068596,  0.13190968,  0.70346046],
		      [ 0.56340401,  0.21963272, -0.79628203,  0.01650193],
		      [ 0.52401101, -0.56833625,  0.22628177,  0.59262376]])

# correct for this X, after SVD calculation:          
explained_var = array([0.44388322,  0.32759266,  0.17262798,  0.05589614])

# Correlation Loadings with Scores of PCA (svd):
Correlation_Loadings = array([[ 0.8398842,   0.51900196,  0.72658209, -0.52234359],
                              [-0.11838866,  0.79063804,  0.15099889,  0.80526119],
                              [ 0.4681721,   0.1825083,  -0.66168686,  0.01371262],
                              [ 0.24777719, -0.26873626,  0.10699672,  0.28022054]])


# amount of decimals to check
accurate = 6   # used where there should be high accuracy
not_so_accurate = 2    

class TestPCA(unittest.TestCase):   
    def test_centering(self):
        # if mean center fails, PCA will fail
        X_c = mean_center(X)
        
        self.failUnlessEqual(X_c.shape, X_centered.shape, 'wrong shape')
        
        for i in range(len(X_centered)):
            for j in range(len(X_centered[0])):
                self.failUnlessAlmostEqual(X_c[i,j], X_centered[i,j], accurate,'wrong value in X_c[%i,%i] (svd)' % (i,j))        
     
    def test_standardization(self):
        # if standardization fails, PCA will fail
        X_std = standardization(X)
        
        self.failUnlessEqual(X_std.shape, X_standardized.shape, 'wrong shape')
        
        for i in range(len(X_standardized)):
            for j in range(len(X_standardized[0])):
                self.failUnlessAlmostEqual(X_std[i,j], X_standardized[i,j], accurate, 'wrong value in X_std[%i,%i] (svd)' % (i,j))        
         
    def test_nipals1a(self):
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, e_var = PCA_nipals(X)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        self.failUnlessEqual(e_var.shape, explained_var.shape, 'wrong shape')
        
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i] (svd)' % (i,j))
        
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i] (svd)' % (i,j))        
        
        for i in range(len(explained_var)):
            self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+'] (svd)')
    
    def test_nipals1b(self):
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, E = PCA_nipals(X, E_matrices=True)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        #self.failUnlessEqual(E.shape, explained_var.shape, 'wrong shape')
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i]' % (i,j))
        
        #print P
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i]' % (i,j))        
        
        #print E
        for i in range(len(explained_var)):
            pass
            #self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+']')



    def test_nipals2(self):
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, e_var = PCA_nipals2(X)        
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        self.failUnlessEqual(e_var.shape, explained_var.shape, 'wrong shape')
        
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i] (svd)' % (i,j))
        
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i] (svd)' % (i,j))        
        
        for i in range(len(explained_var)):
            self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+'] (svd)')
   
   
    def test_nipals2b(self):
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, E = PCA_nipals2(X, E_matrices=True)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        #self.failUnlessEqual(E.shape, explained_var.shape, 'wrong shape')
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i]' % (i,j))
        
        #print P
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i]' % (i,j))        
        
        #print E
        for i in range(len(explained_var)):
            pass
            #self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+']')
  
  
    def test_nipals_c(self):
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, E = PCA_nipals_c(X, E_matrices=True)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        #self.failUnlessEqual(e_var.shape, explained_var.shape, 'wrong shape')
        
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i] (svd)' % (i,j))
        
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i] (svd)' % (i,j))        
        
        for i in range(len(explained_var)):
            pass
            #self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+'] (svd)')
  

    def test_nipals_c2(self):
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, e_var = PCA_nipals_c(X)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        self.failUnlessEqual(e_var.shape, explained_var.shape, 'wrong shape')
        
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i] (svd)' % (i,j))
        
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i] (svd)' % (i,j))        
        
        for i in range(len(explained_var)):
            self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+'] (svd)')
  
  
    def test_svd(self):
        # using default parameters (should be: standardize=True)
        T, P, e_var = PCA_svd(X)       
        self.failUnlessEqual(T.shape, Scores_svd.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_svd.shape, 'wrong shape')
        self.failUnlessEqual(e_var.shape, explained_var.shape, 'wrong shape')
        
        for i in range(len(Scores_svd)):
            for j in range(len(Scores_svd[0])):
                self.failUnlessAlmostEqual(T[i,j], Scores_svd[i,j], accurate, 'wrong value in T[%i,%i] (svd)' % (i,j))
        
        for i in range(len(Loadings_svd)):
            for j in range(len(Loadings_svd[0])):
                self.failUnlessAlmostEqual(P[i,j], Loadings_svd[i,j], accurate, 'wrong value in P[%i,%i] (svd)' % (i,j))        
        
        for i in range(len(explained_var)):
            self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+'] (svd)')
        
        
    def test_corr_loadings(self):
        T, P, e_var = PCA_svd(X)
        CorrLoad = CorrelationLoadings(X, T)
        
        for i in range(len(Correlation_Loadings)):
            for j in range(len(Correlation_Loadings[0])):
                self.failUnlessAlmostEqual(CorrLoad[i,j], Correlation_Loadings[i,j], accurate, 'wrong value in CorrLoad[%i,%i] (svd)' % (i,j))        
           
        
        
############## SPEED TESTING ############## 




def speed_test():

        # averaged Cheese data set:
	Cheese = {}
	Cheese[1] = array([[ 7.1 ,  2.7 ,  4.95,  5.35,  1.  ,  4.5 ,  6.75,  3.2 ,  3.9 ,
		 5.2 ,  4.6 ,  5.15,  4.65,  4.9 ,  1.  ,  4.65,  4.9 ],
	       [ 6.95,  1.95,  5.8 ,  5.65,  1.  ,  5.65,  7.15,  2.35,  3.45,
		 5.05,  5.1 ,  5.9 ,  5.2 ,  5.1 ,  1.  ,  5.7 ,  5.  ],
	       [ 6.55,  4.  ,  3.4 ,  2.65,  1.  ,  2.4 ,  6.3 ,  4.  ,  3.75,
		 5.  ,  5.  ,  3.35,  4.1 ,  3.15,  1.  ,  2.2 ,  4.55],
	       [ 6.95,  2.65,  5.35,  4.75,  1.  ,  4.9 ,  7.05,  2.55,  3.4 ,
		 5.3 ,  5.25,  5.65,  4.4 ,  4.5 ,  1.  ,  4.55,  5.1 ],
	       [ 7.3 ,  1.75,  6.  ,  5.35,  1.  ,  6.2 ,  7.15,  1.7 ,  3.45,
		 5.8 ,  5.75,  6.1 ,  4.95,  4.95,  1.  ,  6.6 ,  5.1 ],
	       [ 6.4 ,  4.05,  4.1 ,  2.95,  1.  ,  2.8 ,  6.15,  3.85,  4.  ,
		 5.1 ,  5.  ,  4.15,  4.55,  3.6 ,  1.  ,  3.15,  4.65],
	       [ 7.8 ,  1.6 ,  5.75,  5.65,  1.  ,  6.35,  7.8 ,  1.  ,  3.75,
		 5.15,  5.15,  6.05,  5.05,  5.2 ,  1.  ,  7.8 ,  4.9 ],
	       [ 7.8 ,  1.  ,  5.9 ,  5.5 ,  1.  ,  7.55,  7.95,  1.  ,  3.45,
		 5.3 ,  5.7 ,  5.8 ,  5.05,  5.15,  1.  ,  7.65,  4.8 ],
	       [ 6.55,  3.8 ,  3.65,  2.5 ,  1.  ,  2.15,  6.45,  3.4 ,  3.55,
		 4.9 ,  4.75,  4.65,  4.3 ,  4.35,  1.  ,  3.15,  4.55],
	       [ 6.75,  4.05,  3.85,  2.75,  1.9 ,  1.7 ,  6.65,  3.2 ,  3.3 ,
		 5.  ,  4.85,  3.95,  3.6 ,  2.7 ,  1.95,  2.05,  4.6 ],
	       [ 6.75,  4.25,  4.15,  3.35,  1.  ,  1.8 ,  6.55,  3.85,  4.05,
		 5.2 ,  5.3 ,  4.65,  4.15,  3.6 ,  1.  ,  2.9 ,  4.95],
	       [ 7.7 ,  1.  ,  5.9 ,  5.65,  1.  ,  6.9 ,  7.65,  1.  ,  3.65,
		 5.05,  4.95,  6.15,  5.05,  5.5 ,  1.  ,  7.4 ,  5.15],
	       [ 7.  ,  6.  ,  1.7 ,  1.  ,  1.  ,  1.  ,  6.65,  5.8 ,  4.2 ,
		 5.45,  4.45,  1.6 ,  3.5 ,  1.  ,  1.  ,  1.  ,  5.15],
	       [ 5.8 ,  5.35,  1.  ,  1.  ,  1.  ,  1.  ,  6.15,  5.95,  4.1 ,
		 6.05,  4.95,  1.  ,  3.35,  1.  ,  1.  ,  1.  ,  4.75]])

	Cheese[2] = array([[ 6.75,  1.  ,  4.65,  3.1 ,  2.6 ,  6.8 ,  7.7 ,  1.  ,  2.6 ,
		 5.3 ,  5.5 ,  4.5 ,  2.6 ,  2.75,  2.4 ,  6.65,  2.4 ],
	       [ 7.15,  1.  ,  4.75,  2.75,  1.65,  5.25,  6.7 ,  1.  ,  2.  ,
		 4.9 ,  5.5 ,  4.55,  1.95,  2.6 ,  1.65,  3.65,  2.8 ],
	       [ 6.05,  1.  ,  4.45,  3.9 ,  2.45,  7.65,  7.15,  1.  ,  2.25,
		 5.9 ,  5.9 ,  4.55,  3.5 ,  3.45,  2.9 ,  8.2 ,  3.25],
	       [ 5.3 ,  1.75,  4.35,  3.1 ,  1.  ,  1.  ,  6.15,  1.6 ,  2.05,
		 5.85,  5.8 ,  3.95,  1.9 ,  3.35,  1.  ,  1.  ,  4.85],
	       [ 5.6 ,  1.  ,  5.  ,  2.95,  1.15,  2.8 ,  6.55,  1.  ,  2.25,
		 6.4 ,  5.95,  5.65,  2.3 ,  3.  ,  1.5 ,  3.3 ,  3.05],
	       [ 5.95,  1.  ,  4.9 ,  2.75,  1.7 ,  3.7 ,  6.95,  1.  ,  2.15,
		 4.6 ,  5.85,  5.1 ,  2.45,  2.35,  1.55,  3.75,  4.15],
	       [ 8.1 ,  1.  ,  4.65,  2.95,  2.7 ,  9.  ,  7.95,  1.  ,  2.25,
		 5.5 ,  6.2 ,  4.4 ,  3.4 ,  4.1 ,  2.45,  9.  ,  1.85],
	       [ 7.6 ,  1.  ,  3.85,  3.8 ,  2.45,  8.45,  7.45,  1.  ,  2.3 ,
		 5.05,  6.45,  4.35,  2.6 ,  3.9 ,  3.  ,  8.4 ,  2.95],
	       [ 6.95,  1.  ,  5.4 ,  3.3 ,  2.  ,  5.5 ,  7.4 ,  1.  ,  2.05,
		 4.8 ,  5.85,  5.15,  2.95,  2.95,  2.35,  6.65,  2.4 ],
	       [ 5.5 ,  1.  ,  5.2 ,  3.1 ,  1.45,  1.  ,  5.9 ,  1.  ,  2.  ,
		 4.85,  5.3 ,  4.9 ,  2.1 ,  3.  ,  1.  ,  1.  ,  3.35],
	       [ 6.4 ,  1.  ,  4.9 ,  2.9 ,  1.7 ,  3.4 ,  6.55,  1.  ,  2.1 ,
		 4.9 ,  5.6 ,  4.9 ,  2.8 ,  3.95,  1.65,  3.85,  3.4 ],
	       [ 7.35,  1.  ,  3.85,  3.15,  2.6 ,  9.  ,  8.4 ,  1.  ,  1.95,
		 4.5 ,  6.15,  3.9 ,  3.5 ,  3.65,  3.4 ,  9.  ,  2.2 ],
	       [ 5.45,  5.15,  1.55,  1.  ,  1.  ,  1.  ,  6.35,  5.  ,  2.35,
		 5.9 ,  5.9 ,  1.95,  1.9 ,  1.  ,  1.  ,  1.  ,  2.9 ],
	       [ 5.6 ,  4.8 ,  1.5 ,  1.  ,  1.  ,  1.  ,  6.15,  4.95,  1.95,
		 6.45,  5.4 ,  2.2 ,  1.4 ,  1.  ,  1.  ,  1.  ,  3.35]])

	Cheese[3] = array([[ 6.35,  4.55,  2.65,  2.  ,  6.8 ,  3.5 ,  2.6 ,  5.5 ,  6.25,
		 3.4 ,  3.2 ,  4.15,  5.  ],
	       [ 6.5 ,  4.45,  2.6 ,  3.5 ,  6.65,  3.95,  2.35,  4.85,  6.5 ,
		 2.4 ,  2.95,  3.85,  4.7 ],
	       [ 6.6 ,  4.2 ,  3.8 ,  1.  ,  7.  ,  2.6 ,  2.65,  5.65,  6.35,
		 4.25,  3.05,  4.05,  4.7 ],
	       [ 5.3 ,  5.3 ,  2.9 ,  2.1 ,  6.7 ,  4.5 ,  2.4 ,  4.95,  6.35,
		 3.35,  2.65,  2.5 ,  4.85],
	       [ 7.4 ,  1.8 ,  4.5 ,  5.8 ,  7.95,  1.7 ,  1.8 ,  5.6 ,  6.5 ,
		 3.8 ,  3.1 ,  5.65,  4.55],
	       [ 5.8 ,  6.1 ,  1.45,  1.  ,  6.35,  5.45,  2.85,  4.6 ,  5.1 ,
		 1.55,  2.1 ,  1.  ,  4.9 ],
	       [ 6.75,  4.6 ,  4.05,  2.55,  7.15,  2.85,  2.5 ,  4.6 ,  6.3 ,
		 4.5 ,  3.15,  5.2 ,  4.85],
	       [ 5.7 ,  4.55,  3.55,  1.  ,  6.4 ,  4.05,  2.3 ,  4.6 ,  6.35,
		 3.75,  2.95,  2.05,  5.1 ],
	       [ 6.1 ,  4.75,  2.3 ,  1.  ,  6.75,  4.05,  2.8 ,  5.1 ,  6.1 ,
		 3.  ,  2.75,  2.5 ,  4.9 ],
	       [ 6.3 ,  3.65,  3.8 ,  1.  ,  6.9 ,  3.2 ,  2.4 ,  5.  ,  6.7 ,
		 4.15,  3.15,  3.8 ,  4.95],
	       [ 6.45,  4.2 ,  2.9 ,  3.  ,  6.85,  3.65,  2.1 ,  4.9 ,  6.2 ,
		 2.75,  1.95,  3.9 ,  4.8 ],
	       [ 6.75,  3.8 ,  3.25,  1.85,  7.3 ,  1.8 ,  2.15,  5.  ,  6.9 ,
		 3.25,  3.1 ,  4.9 ,  4.85],
	       [ 5.75,  5.75,  1.  ,  1.  ,  6.8 ,  4.75,  2.45,  5.6 ,  6.6 ,
		 3.95,  3.  ,  1.  ,  4.95],
	       [ 5.65,  5.65,  1.5 ,  1.  ,  6.75,  4.  ,  1.95,  5.65,  6.8 ,
		 3.2 ,  2.8 ,  1.  ,  5.3 ]])

	Cheese[4] = array([[ 6.5 ,  3.85,  5.  ,  2.7 ,  1.3 ,  2.55,  6.6 ,  4.7 ,  3.9 ,
		 6.55,  6.  ,  3.15,  3.4 ,  1.55,  1.  ,  1.6 ,  4.8 ],
	       [ 6.3 ,  3.8 ,  5.5 ,  2.35,  1.4 ,  1.7 ,  7.05,  4.05,  4.  ,
		 6.8 ,  5.6 ,  5.55,  4.3 ,  1.9 ,  1.  ,  3.45,  4.75],
	       [ 6.25,  3.85,  2.95,  1.75,  2.05,  1.  ,  6.55,  4.55,  3.5 ,
		 6.1 ,  5.5 ,  3.2 ,  3.75,  1.45,  1.  ,  1.25,  4.8 ],
	       [ 6.65,  3.25,  5.45,  2.7 ,  1.  ,  2.5 ,  6.8 ,  3.8 ,  4.3 ,
		 5.8 ,  6.25,  5.1 ,  4.15,  2.65,  1.  ,  2.65,  4.85],
	       [ 6.65,  3.6 ,  5.8 ,  2.5 ,  1.  ,  3.95,  7.05,  3.5 ,  4.05,
		 7.2 ,  6.1 ,  4.75,  4.15,  1.9 ,  1.  ,  2.6 ,  4.5 ],
	       [ 6.5 ,  5.75,  3.1 ,  2.25,  1.  ,  2.15,  6.45,  6.45,  3.95,
		 5.5 ,  5.25,  3.05,  3.65,  2.2 ,  1.  ,  2.15,  4.55],
	       [ 6.25,  6.3 ,  2.05,  1.  ,  1.  ,  1.  ,  6.5 ,  5.5 ,  3.85,
		 5.75,  5.65,  3.15,  3.2 ,  2.05,  1.  ,  1.  ,  4.65],
	       [ 7.4 ,  1.95,  7.95,  2.25,  1.  ,  5.2 ,  7.6 ,  2.5 ,  3.8 ,
		 5.55,  5.95,  6.4 ,  4.4 ,  2.55,  1.  ,  4.15,  4.6 ],
	       [ 6.45,  3.85,  5.05,  2.65,  1.5 ,  2.  ,  6.8 ,  3.65,  3.75,
		 5.95,  5.35,  5.3 ,  3.8 ,  1.95,  1.  ,  3.05,  4.45],
	       [ 6.05,  4.55,  3.35,  2.  ,  1.  ,  1.35,  6.95,  4.8 ,  4.2 ,
		 5.75,  6.1 ,  4.1 ,  3.9 ,  2.  ,  1.4 ,  2.  ,  4.45],
	       [ 6.1 ,  3.85,  5.25,  1.  ,  1.  ,  2.35,  7.1 ,  3.  ,  4.1 ,
		 5.9 ,  6.05,  6.  ,  4.45,  2.05,  1.  ,  4.6 ,  4.3 ],
	       [ 6.5 ,  3.4 ,  5.55,  2.75,  1.2 ,  3.3 ,  7.2 ,  2.25,  3.6 ,
		 5.45,  5.5 ,  6.7 ,  4.25,  2.95,  1.  ,  4.6 ,  4.8 ],
	       [ 6.5 ,  7.25,  1.  ,  1.  ,  1.  ,  1.  ,  7.05,  7.9 ,  4.4 ,
		 6.2 ,  5.1 ,  1.  ,  2.9 ,  1.  ,  1.  ,  1.  ,  4.8 ],
	       [ 6.5 ,  5.1 ,  4.1 ,  1.  ,  1.  ,  1.25,  6.9 ,  4.1 ,  4.1 ,
		 7.5 ,  5.65,  3.9 ,  4.1 ,  1.  ,  1.  ,  1.3 ,  4.8 ]])

	Cheese[5] = array([[ 6.85,  3.9 ,  3.75,  2.05,  1.7 ,  3.45,  7.2 ,  3.3 ,  2.25,
		 5.15,  5.1 ,  4.  ,  4.  ,  2.15,  1.65,  3.5 ,  3.25],
	       [ 6.9 ,  2.75,  4.15,  1.6 ,  1.  ,  3.75,  7.05,  2.7 ,  2.65,
		 4.8 ,  4.6 ,  4.  ,  3.45,  1.8 ,  1.45,  3.85,  3.1 ],
	       [ 6.4 ,  2.35,  4.9 ,  2.4 ,  1.  ,  5.05,  7.3 ,  1.75,  2.2 ,
		 4.9 ,  5.6 ,  5.15,  4.7 ,  2.65,  1.9 ,  5.9 ,  3.55],
	       [ 7.15,  2.35,  4.65,  2.2 ,  1.75,  4.85,  7.6 ,  1.85,  2.15,
		 4.6 ,  5.45,  5.25,  4.1 ,  2.2 ,  2.25,  5.85,  3.25],
	       [ 6.35,  3.5 ,  3.6 ,  1.4 ,  2.05,  3.4 ,  6.85,  3.25,  2.3 ,
		 4.7 ,  4.4 ,  3.75,  3.55,  1.65,  2.15,  3.3 ,  3.15],
	       [ 7.05,  2.6 ,  4.5 ,  1.85,  4.25,  4.  ,  7.2 ,  2.55,  2.75,
		 5.05,  4.8 ,  4.45,  3.7 ,  1.6 ,  2.6 ,  4.05,  3.25],
	       [ 7.2 ,  2.65,  4.75,  2.25,  4.2 ,  4.15,  7.45,  2.6 ,  2.55,
		 5.1 ,  5.6 ,  5.15,  5.05,  1.8 ,  3.3 ,  4.55,  3.7 ],
	       [ 7.  ,  2.45,  4.55,  2.25,  1.5 ,  4.7 ,  7.1 ,  2.55,  2.65,
		 4.55,  4.45,  4.45,  3.35,  1.5 ,  1.45,  4.75,  3.1 ],
	       [ 6.65,  3.5 ,  3.7 ,  1.85,  1.  ,  2.15,  6.35,  3.7 ,  3.05,
		 4.65,  4.15,  3.5 ,  3.55,  1.  ,  1.  ,  2.1 ,  3.1 ],
	       [ 6.55,  3.8 ,  3.1 ,  1.  ,  1.  ,  2.1 ,  6.5 ,  3.55,  3.05,
		 4.5 ,  3.7 ,  3.05,  3.15,  1.  ,  1.  ,  2.25,  3.15],
	       [ 6.7 ,  3.5 ,  3.65,  1.65,  1.4 ,  3.2 ,  6.95,  2.95,  2.45,
		 4.95,  4.85,  4.25,  4.05,  1.75,  1.5 ,  4.25,  3.15],
	       [ 7.  ,  2.85,  3.15,  2.05,  2.  ,  3.35,  7.45,  2.3 ,  3.25,
		 5.1 ,  5.35,  4.8 ,  4.15,  1.75,  2.65,  5.2 ,  3.1 ],
	       [ 6.5 ,  4.2 ,  3.15,  1.  ,  1.  ,  1.55,  6.75,  3.25,  2.5 ,
		 4.7 ,  4.25,  3.4 ,  3.85,  1.75,  1.  ,  2.8 ,  3.05],
	       [ 6.3 ,  5.  ,  2.2 ,  1.  ,  1.  ,  1.  ,  6.15,  5.45,  2.25,
		 4.6 ,  2.75,  1.45,  3.35,  1.  ,  1.  ,  1.  ,  3.1 ]])

	Cheese[6] = array([[ 5.5 ,  2.85,  3.2 ,  1.45,  1.  ,  4.15,  6.65,  2.6 ,  1.  ,
		 5.75,  4.2 ,  2.4 ,  2.5 ,  1.  ,  3.45,  4.75],
	       [ 6.3 ,  1.  ,  3.9 ,  1.  ,  1.55,  7.3 ,  5.7 ,  1.  ,  1.  ,
		 4.75,  3.4 ,  3.65,  2.6 ,  1.  ,  5.9 ,  4.65],
	       [ 6.  ,  4.5 ,  3.8 ,  1.  ,  1.  ,  4.3 ,  6.4 ,  3.95,  1.  ,
		 5.7 ,  4.3 ,  2.2 ,  2.8 ,  1.  ,  2.15,  4.9 ],
	       [ 5.65,  1.35,  3.95,  1.  ,  1.  ,  7.1 ,  5.5 ,  2.  ,  1.  ,
		 4.25,  3.  ,  2.95,  2.6 ,  1.  ,  5.5 ,  4.85],
	       [ 6.35,  1.  ,  4.  ,  1.  ,  1.  ,  7.45,  5.9 ,  2.05,  1.  ,
		 5.3 ,  3.95,  3.35,  2.65,  1.  ,  7.1 ,  5.05],
	       [ 5.85,  1.9 ,  3.25,  2.15,  1.  ,  4.8 ,  6.15,  3.5 ,  1.4 ,
		 4.65,  3.75,  1.65,  2.3 ,  1.  ,  2.15,  4.5 ],
	       [ 5.8 ,  2.9 ,  3.15,  1.  ,  1.  ,  4.55,  5.8 ,  2.2 ,  1.  ,
		 4.7 ,  3.65,  3.45,  2.35,  1.  ,  4.  ,  5.  ],
	       [ 5.9 ,  2.6 ,  3.85,  1.  ,  1.  ,  5.05,  5.95,  1.6 ,  1.  ,
		 5.1 ,  4.05,  3.5 ,  2.9 ,  1.6 ,  5.7 ,  4.55],
	       [ 6.15,  4.35,  3.25,  1.  ,  1.  ,  3.95,  5.9 ,  2.7 ,  1.  ,
		 5.15,  4.  ,  3.4 ,  2.4 ,  1.  ,  4.5 ,  4.9 ],
	       [ 5.8 ,  4.8 ,  2.85,  1.  ,  1.  ,  2.2 ,  5.9 ,  2.6 ,  1.  ,
		 5.  ,  3.6 ,  3.2 ,  2.35,  1.  ,  3.2 ,  4.6 ],
	       [ 6.1 ,  1.  ,  4.25,  1.  ,  1.  ,  6.7 ,  6.15,  1.5 ,  1.  ,
		 4.5 ,  3.65,  3.65,  2.65,  1.  ,  6.65,  5.05],
	       [ 5.9 ,  1.  ,  4.1 ,  1.65,  1.  ,  7.75,  6.  ,  1.  ,  1.  ,
		 5.5 ,  4.35,  3.7 ,  2.3 ,  1.6 ,  7.75,  5.05],
	       [ 6.15,  5.7 ,  1.  ,  1.  ,  1.  ,  1.  ,  6.35,  5.3 ,  1.4 ,
		 5.1 ,  3.65,  1.  ,  2.35,  1.  ,  1.  ,  4.85],
	       [ 5.6 ,  5.6 ,  1.75,  1.  ,  1.  ,  1.  ,  6.  ,  5.35,  1.  ,
		 5.35,  3.9 ,  1.  ,  2.8 ,  1.  ,  1.  ,  5.15]])

	Cheese[7] = array([[ 4.55,  2.15,  1.7 ,  1.  ,  1.  ,  1.  ,  5.55,  2.8 ,  2.15,
		 3.55,  2.4 ,  2.55,  2.05,  1.  ,  1.  ,  1.  ,  4.05],
	       [ 5.1 ,  1.7 ,  2.8 ,  1.95,  1.6 ,  1.7 ,  5.8 ,  1.75,  1.9 ,
		 4.1 ,  2.7 ,  3.6 ,  2.25,  1.  ,  1.45,  3.35,  4.2 ],
	       [ 5.5 ,  2.2 ,  2.8 ,  1.8 ,  1.  ,  1.45,  5.7 ,  2.8 ,  1.95,
		 3.75,  3.5 ,  3.35,  2.15,  1.  ,  1.  ,  1.3 ,  4.5 ],
	       [ 4.95,  2.1 ,  2.5 ,  1.  ,  1.  ,  1.75,  5.65,  2.25,  1.9 ,
		 3.65,  3.2 ,  2.45,  2.5 ,  1.  ,  1.05,  2.15,  4.15],
	       [ 6.5 ,  1.  ,  3.65,  1.4 ,  1.  ,  4.7 ,  6.05,  1.  ,  1.9 ,
		 3.9 ,  3.7 ,  3.6 ,  2.75,  1.75,  1.  ,  5.1 ,  4.15],
	       [ 5.4 ,  1.75,  2.95,  1.5 ,  1.3 ,  2.  ,  5.85,  2.25,  2.25,
		 4.35,  3.15,  3.55,  2.25,  1.6 ,  1.35,  2.6 ,  4.35],
	       [ 5.4 ,  1.5 ,  3.25,  1.6 ,  1.4 ,  2.7 ,  6.05,  1.4 ,  2.1 ,
		 2.75,  3.6 ,  3.1 ,  2.25,  1.05,  1.75,  3.2 ,  4.  ],
	       [ 5.3 ,  1.8 ,  3.25,  1.4 ,  1.  ,  1.45,  6.  ,  1.4 ,  1.9 ,
		 3.45,  3.45,  3.7 ,  3.45,  2.  ,  1.7 ,  3.25,  3.85],
	       [ 4.85,  2.15,  2.  ,  1.4 ,  1.  ,  1.  ,  5.6 ,  2.1 ,  2.2 ,
		 3.7 ,  2.85,  3.4 ,  2.55,  1.35,  1.  ,  1.6 ,  4.  ],
	       [ 4.75,  1.4 ,  2.5 ,  1.9 ,  1.  ,  1.  ,  5.7 ,  1.5 ,  2.1 ,
		 3.65,  2.85,  3.15,  1.9 ,  1.  ,  1.35,  3.2 ,  4.  ],
	       [ 5.4 ,  2.6 ,  3.15,  1.55,  1.45,  1.  ,  5.7 ,  2.6 ,  2.15,
		 4.25,  3.05,  2.9 ,  2.75,  1.  ,  1.25,  2.4 ,  3.95],
	       [ 5.55,  1.  ,  3.75,  2.1 ,  1.  ,  4.2 ,  6.05,  1.4 ,  2.2 ,
		 3.55,  3.4 ,  3.85,  2.65,  1.75,  1.35,  4.45,  4.1 ],
	       [ 5.5 ,  3.15,  2.15,  1.  ,  1.  ,  1.  ,  5.7 ,  3.8 ,  2.2 ,
		 3.6 ,  3.35,  2.2 ,  1.6 ,  1.  ,  1.  ,  1.  ,  3.7 ],
	       [ 5.15,  2.5 ,  1.4 ,  1.  ,  1.  ,  1.  ,  5.7 ,  3.2 ,  2.1 ,
		 4.15,  2.85,  1.45,  2.1 ,  1.  ,  1.  ,  1.  ,  4.2 ]])

	Cheese[8] = array([[ 5.75,  3.5 ,  3.6 ,  1.6 ,  2.5 ,  6.55,  3.2 ,  2.95,  5.75,
		 3.95,  3.25,  3.  ,  1.65,  1.  ,  2.35,  4.55],
	       [ 5.75,  4.3 ,  3.5 ,  1.75,  1.4 ,  5.95,  4.9 ,  3.05,  5.5 ,
		 4.1 ,  3.55,  2.75,  1.55,  1.6 ,  1.5 ,  4.75],
	       [ 5.85,  4.1 ,  2.8 ,  1.85,  1.85,  6.15,  4.75,  2.7 ,  4.9 ,
		 3.4 ,  3.25,  2.7 ,  1.  ,  1.  ,  2.05,  4.65],
	       [ 6.3 ,  2.75,  4.5 ,  2.4 ,  2.85,  6.45,  3.55,  2.8 ,  4.7 ,
		 4.05,  4.7 ,  3.1 ,  1.2 ,  1.  ,  3.  ,  4.65],
	       [ 6.85,  1.  ,  5.5 ,  4.15,  4.95,  7.8 ,  1.  ,  2.  ,  4.95,
		 5.35,  5.75,  2.7 ,  3.9 ,  1.  ,  5.35,  4.3 ],
	       [ 6.1 ,  3.7 ,  3.75,  2.25,  2.2 ,  6.1 ,  4.2 ,  3.1 ,  4.7 ,
		 3.3 ,  3.8 ,  2.5 ,  1.25,  1.  ,  1.8 ,  5.05],
	       [ 6.85,  1.65,  5.45,  3.6 ,  3.95,  7.6 ,  1.  ,  2.4 ,  4.7 ,
		 4.55,  5.3 ,  3.8 ,  3.6 ,  1.  ,  4.6 ,  4.3 ],
	       [ 6.25,  3.35,  4.9 ,  1.55,  1.9 ,  6.15,  3.9 ,  2.5 ,  4.85,
		 4.05,  4.5 ,  3.3 ,  1.55,  1.  ,  1.95,  4.45],
	       [ 5.95,  3.85,  3.55,  2.25,  2.15,  6.3 ,  3.35,  2.95,  4.85,
		 3.95,  3.7 ,  3.3 ,  1.85,  1.  ,  2.35,  4.3 ],
	       [ 5.4 ,  4.05,  3.55,  1.3 ,  1.55,  5.6 ,  3.9 ,  2.7 ,  4.6 ,
		 3.35,  4.25,  2.75,  1.3 ,  1.  ,  1.55,  4.35],
	       [ 6.  ,  3.45,  4.45,  2.85,  3.25,  7.3 ,  1.  ,  2.3 ,  4.75,
		 4.55,  5.  ,  3.  ,  3.8 ,  1.  ,  4.95,  4.35],
	       [ 6.15,  2.1 ,  4.9 ,  3.25,  3.2 ,  6.5 ,  2.05,  2.5 ,  4.7 ,
		 4.  ,  4.85,  3.2 ,  2.6 ,  1.  ,  3.75,  4.4 ],
	       [ 5.75,  5.75,  1.5 ,  1.  ,  1.  ,  5.6 ,  6.  ,  3.45,  5.4 ,
		 3.5 ,  1.5 ,  2.75,  1.  ,  1.  ,  1.  ,  5.  ],
	       [ 5.95,  4.45,  3.1 ,  1.  ,  1.45,  6.1 ,  4.85,  2.45,  6.2 ,
		 4.3 ,  2.3 ,  2.25,  1.  ,  1.  ,  1.  ,  5.1 ]])

	Cheese[9] = array([[ 6.4 ,  5.45,  1.4 ,  1.  ,  1.  ,  1.  ,  6.5 ,  4.65,  2.55,
		 3.15,  3.15,  1.65,  1.  ,  1.  ,  1.  ,  1.  ,  2.9 ],
	       [ 8.  ,  1.55,  5.9 ,  4.65,  1.  ,  4.9 ,  8.05,  1.45,  1.4 ,
		 3.1 ,  5.25,  3.95,  2.35,  4.65,  3.45,  4.1 ,  2.7 ],
	       [ 6.9 ,  4.7 ,  1.85,  1.9 ,  1.  ,  1.7 ,  6.9 ,  5.15,  3.05,
		 3.1 ,  3.05,  1.65,  1.  ,  1.55,  1.55,  1.6 ,  2.5 ],
	       [ 7.  ,  4.85,  1.95,  1.  ,  1.  ,  1.  ,  6.65,  4.75,  2.5 ,
		 2.85,  2.8 ,  1.9 ,  1.  ,  1.  ,  1.4 ,  1.  ,  2.05],
	       [ 7.3 ,  3.05,  4.4 ,  2.1 ,  1.  ,  2.15,  7.75,  3.45,  2.1 ,
		 3.8 ,  5.05,  3.9 ,  2.15,  3.05,  2.45,  3.  ,  2.4 ],
	       [ 6.5 ,  4.2 ,  1.9 ,  1.65,  1.  ,  1.6 ,  6.3 ,  5.05,  2.4 ,
		 2.7 ,  2.7 ,  1.65,  1.  ,  1.7 ,  1.45,  2.  ,  2.95],
	       [ 7.95,  1.  ,  5.1 ,  4.15,  2.  ,  3.5 ,  7.7 ,  1.  ,  1.85,
		 2.95,  5.15,  4.9 ,  1.  ,  3.75,  3.05,  4.5 ,  2.55],
	       [ 7.35,  4.1 ,  3.  ,  2.1 ,  1.  ,  2.15,  7.55,  3.8 ,  1.75,
		 2.2 ,  4.2 ,  3.95,  2.3 ,  3.55,  3.  ,  3.  ,  2.65],
	       [ 7.5 ,  1.  ,  3.9 ,  3.55,  1.  ,  3.25,  6.95,  1.  ,  2.6 ,
		 3.75,  4.8 ,  4.4 ,  2.35,  4.45,  3.8 ,  4.25,  2.75],
	       [ 6.65,  2.8 ,  2.95,  1.  ,  1.  ,  1.  ,  6.7 ,  4.6 ,  2.3 ,
		 2.75,  2.75,  2.1 ,  1.  ,  1.  ,  1.  ,  1.  ,  2.6 ],
	       [ 6.75,  3.05,  3.6 ,  3.2 ,  1.  ,  2.8 ,  7.7 ,  1.8 ,  1.75,
		 2.85,  5.5 ,  4.45,  2.6 ,  4.9 ,  4.35,  4.6 ,  2.3 ],
	       [ 7.85,  1.  ,  4.75,  4.95,  2.1 ,  3.95,  7.7 ,  1.  ,  1.95,
		 3.65,  4.9 ,  3.7 ,  2.7 ,  5.  ,  3.85,  3.8 ,  2.15],
	       [ 7.1 ,  4.6 ,  1.25,  1.65,  1.  ,  1.6 ,  6.55,  4.8 ,  2.4 ,
		 3.25,  3.65,  1.85,  1.  ,  1.85,  2.65,  1.8 ,  2.55],
	       [ 6.65,  5.5 ,  1.85,  1.  ,  1.  ,  1.  ,  7.15,  5.25,  2.05,
		 3.35,  4.25,  1.35,  1.  ,  1.  ,  1.3 ,  1.  ,  1.65]])

	Cheese[10] = array([[ 4.2 ,  4.15,  1.45,  2.3 ,  1.55,  1.45,  5.05,  3.5 ,  1.8 ,
		 2.5 ,  2.55,  2.35,  2.5 ,  2.9 ,  1.35,  3.5 ,  4.25],
	       [ 4.8 ,  3.  ,  1.65,  2.05,  1.5 ,  2.7 ,  5.45,  2.9 ,  1.6 ,
		 1.9 ,  2.65,  1.3 ,  1.9 ,  1.95,  1.  ,  3.25,  5.05],
	       [ 5.9 ,  4.2 ,  3.45,  1.4 ,  1.  ,  2.7 ,  5.7 ,  3.15,  1.5 ,
		 2.9 ,  3.9 ,  2.65,  2.7 ,  2.1 ,  1.9 ,  4.35,  4.85],
	       [ 5.4 ,  4.1 ,  1.4 ,  1.25,  1.  ,  1.6 ,  6.5 ,  3.05,  1.4 ,
		 2.7 ,  3.75,  1.4 ,  1.55,  1.7 ,  1.  ,  2.95,  4.8 ],
	       [ 6.5 ,  3.55,  2.6 ,  1.65,  1.3 ,  3.5 ,  5.8 ,  2.65,  1.4 ,
		 2.7 ,  3.75,  2.25,  1.9 ,  1.8 ,  1.  ,  4.35,  4.85],
	       [ 4.95,  4.35,  1.5 ,  1.  ,  1.  ,  1.  ,  4.75,  3.95,  1.75,
		 2.8 ,  3.65,  2.  ,  1.95,  2.5 ,  1.75,  2.7 ,  5.55],
	       [ 5.2 ,  4.2 ,  1.55,  1.  ,  1.  ,  1.  ,  5.6 ,  4.  ,  1.45,
		 3.  ,  3.9 ,  2.25,  2.85,  1.9 ,  1.  ,  2.  ,  4.75],
	       [ 7.  ,  3.95,  2.1 ,  1.85,  1.  ,  2.3 ,  6.35,  3.7 ,  1.65,
		 2.4 ,  3.65,  1.55,  1.25,  2.2 ,  1.4 ,  4.9 ,  4.15],
	       [ 7.  ,  4.85,  1.9 ,  2.05,  1.  ,  1.9 ,  6.7 ,  3.8 ,  1.6 ,
		 3.85,  4.55,  2.65,  2.9 ,  2.1 ,  1.7 ,  3.75,  4.95],
	       [ 6.2 ,  2.9 ,  2.95,  1.  ,  1.  ,  1.  ,  5.75,  3.2 ,  1.2 ,
		 2.6 ,  4.4 ,  3.15,  1.65,  2.1 ,  1.6 ,  4.9 ,  5.  ],
	       [ 5.7 ,  3.4 ,  2.05,  1.7 ,  1.  ,  2.35,  5.05,  2.1 ,  1.55,
		 1.95,  2.4 ,  1.35,  1.8 ,  2.6 ,  2.3 ,  3.15,  5.1 ],
	       [ 5.5 ,  4.25,  1.9 ,  1.35,  1.  ,  3.2 ,  5.3 ,  3.2 ,  1.3 ,
		 2.2 ,  4.3 ,  1.35,  2.  ,  3.3 ,  1.4 ,  4.35,  5.  ],
	       [ 6.4 ,  4.65,  1.25,  1.  ,  1.  ,  1.  ,  5.45,  3.65,  1.7 ,
		 2.7 ,  2.35,  1.  ,  1.65,  1.8 ,  1.  ,  1.35,  5.05],
	       [ 6.1 ,  4.55,  2.05,  1.  ,  1.  ,  1.  ,  4.85,  3.2 ,  1.5 ,
		 2.55,  2.65,  1.25,  2.1 ,  1.  ,  1.  ,  1.  ,  4.45]])

	Cheese[11] = array([[ 4.05,  2.4 ,  2.25,  1.3 ,  1.25,  1.  ,  4.3 ,  3.15,  1.8 ,
		 3.  ,  1.7 ,  1.  ,  1.  ,  1.2 ,  1.25,  1.2 ,  3.8 ],
	       [ 4.5 ,  2.25,  1.65,  1.  ,  1.  ,  1.  ,  3.45,  2.1 ,  1.3 ,
		 1.95,  1.15,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  3.05],
	       [ 4.9 ,  2.05,  5.1 ,  1.75,  1.  ,  1.55,  4.15,  3.5 ,  1.55,
		 3.2 ,  1.6 ,  2.  ,  1.45,  1.3 ,  1.  ,  1.45,  3.9 ],
	       [ 6.  ,  2.15,  5.1 ,  1.7 ,  1.35,  4.75,  4.4 ,  2.4 ,  1.55,
		 2.35,  1.9 ,  1.8 ,  1.5 ,  1.  ,  1.  ,  1.8 ,  3.2 ],
	       [ 6.1 ,  2.2 ,  4.65,  1.7 ,  1.  ,  5.6 ,  5.15,  2.55,  2.05,
		 3.1 ,  2.9 ,  3.45,  2.05,  1.55,  1.  ,  4.  ,  3.85],
	       [ 4.55,  3.55,  3.45,  1.25,  1.8 ,  2.5 ,  4.1 ,  2.5 ,  1.9 ,
		 2.4 ,  1.8 ,  1.9 ,  1.  ,  1.  ,  1.6 ,  2.05,  3.6 ],
	       [ 4.25,  1.6 ,  4.  ,  1.5 ,  1.5 ,  1.85,  5.1 ,  1.9 ,  1.8 ,
		 3.45,  3.25,  4.55,  2.75,  1.45,  1.9 ,  1.95,  3.5 ],
	       [ 5.9 ,  2.05,  4.75,  1.6 ,  1.4 ,  4.2 ,  5.2 ,  2.2 ,  1.9 ,
		 2.05,  3.7 ,  3.5 ,  1.45,  1.4 ,  1.4 ,  3.45,  3.8 ],
	       [ 4.45,  2.5 ,  3.25,  2.4 ,  1.  ,  1.4 ,  4.2 ,  2.7 ,  2.  ,
		 3.2 ,  2.45,  2.6 ,  1.  ,  1.55,  1.  ,  1.9 ,  3.75],
	       [ 4.8 ,  2.65,  2.9 ,  1.25,  1.2 ,  1.9 ,  4.75,  2.85,  1.85,
		 3.1 ,  2.6 ,  2.65,  1.95,  2.1 ,  1.55,  2.35,  3.8 ],
	       [ 5.6 ,  2.6 ,  4.55,  2.3 ,  1.25,  4.05,  6.  ,  2.55,  1.85,
		 2.65,  3.65,  4.8 ,  2.4 ,  1.45,  1.25,  5.2 ,  3.9 ],
	       [ 5.1 ,  2.55,  4.8 ,  1.45,  2.5 ,  2.85,  5.2 ,  2.05,  2.2 ,
		 3.5 ,  2.05,  4.9 ,  1.65,  2.  ,  2.25,  3.15,  4.8 ],
	       [ 4.35,  2.8 ,  1.2 ,  1.2 ,  1.  ,  1.  ,  4.45,  3.  ,  1.45,
		 2.4 ,  1.15,  1.  ,  1.15,  1.  ,  1.  ,  1.  ,  3.65],
	       [ 4.15,  3.05,  1.  ,  1.3 ,  1.  ,  1.  ,  4.9 ,  3.4 ,  1.65,
		 3.25,  1.35,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  4.1 ]])

	Cheese[12] = array([[ 4.75,  4.5 ,  3.5 ,  2.05,  1.  ,  1.  ,  5.  ,  7.05,  1.25,
		 4.75,  6.55,  2.  ,  1.7 ,  2.8 ,  1.35,  1.4 ,  4.75],
	       [ 4.65,  5.15,  1.4 ,  1.  ,  1.  ,  1.75,  5.05,  5.15,  1.  ,
		 3.5 ,  2.85,  1.  ,  1.35,  1.  ,  1.  ,  1.2 ,  4.3 ],
	       [ 6.5 ,  4.75,  2.95,  1.3 ,  1.  ,  1.9 ,  4.65,  4.3 ,  1.  ,
		 4.  ,  4.1 ,  2.15,  1.3 ,  1.25,  1.  ,  1.85,  4.05],
	       [ 6.4 ,  5.6 ,  4.05,  1.  ,  1.  ,  1.9 ,  4.9 ,  4.45,  1.  ,
		 4.55,  5.8 ,  2.55,  1.5 ,  1.1 ,  1.5 ,  1.05,  5.95],
	       [ 6.8 ,  5.7 ,  3.05,  1.  ,  1.  ,  2.75,  4.55,  5.7 ,  1.  ,
		 5.45,  3.45,  1.  ,  1.25,  1.  ,  1.4 ,  2.1 ,  2.95],
	       [ 6.45,  6.2 ,  4.25,  2.25,  1.  ,  1.75,  5.75,  5.65,  1.35,
		 2.85,  4.8 ,  2.5 ,  1.  ,  2.  ,  1.45,  2.2 ,  5.75],
	       [ 5.7 ,  5.05,  3.6 ,  1.  ,  1.1 ,  1.3 ,  6.  ,  5.8 ,  1.  ,
		 4.4 ,  5.3 ,  1.85,  2.2 ,  1.4 ,  1.95,  2.45,  2.6 ],
	       [ 6.3 ,  5.5 ,  3.35,  1.  ,  1.  ,  1.5 ,  5.  ,  4.7 ,  1.4 ,
		 2.8 ,  5.95,  2.25,  1.2 ,  1.3 ,  1.  ,  1.55,  4.2 ],
	       [ 3.35,  3.65,  1.2 ,  1.25,  1.  ,  1.55,  5.4 ,  5.  ,  1.  ,
		 3.3 ,  7.2 ,  1.  ,  1.25,  1.35,  1.55,  1.7 ,  2.55],
	       [ 5.2 ,  4.6 ,  1.6 ,  1.  ,  1.  ,  1.9 ,  6.  ,  5.9 ,  1.  ,
		 4.35,  3.7 ,  1.4 ,  1.4 ,  1.  ,  1.25,  1.  ,  3.75],
	       [ 4.75,  3.55,  2.7 ,  1.2 ,  1.  ,  1.5 ,  4.3 ,  3.75,  1.25,
		 4.1 ,  3.9 ,  2.05,  1.  ,  1.  ,  1.2 ,  1.15,  4.3 ],
	       [ 4.65,  3.85,  3.6 ,  1.  ,  1.  ,  2.7 ,  6.25,  4.1 ,  1.  ,
		 3.75,  5.15,  2.7 ,  1.  ,  1.2 ,  1.45,  2.45,  3.4 ],
	       [ 3.45,  2.95,  1.2 ,  1.  ,  1.  ,  1.  ,  5.5 ,  4.65,  1.  ,
		 5.5 ,  3.95,  1.  ,  1.  ,  1.25,  1.  ,  1.  ,  3.45],
	       [ 4.35,  4.4 ,  1.7 ,  1.  ,  1.  ,  1.2 ,  5.5 ,  5.6 ,  1.  ,
		 4.55,  4.6 ,  1.45,  1.25,  1.  ,  1.  ,  1.  ,  4.65]]) 
 
 
        
        ost = True
        N = 1000
        M = 1000
        times = 10
        t = 0.0001
        PCs = 4
        
        r = zeros((M, N), float)
        for i in range(0, M):
            r[i] = random.rand(N)
        

	if ost:
	    X = Cheese
	else:
            X = r
        """
	print 
	print "nipals numeric array (5x)"
	c0 = time.clock()
	#T1, P1, explained_var1 = PCA_nipals(Ost, False)
	#T1, P1, explained_var1 = PCA_nipals(Ost, False)
	#T1, P1, explained_var1 = PCA_nipals(Ost, False)
	#T1, P1, explained_var1 = PCA_nipals(Ost, False)
	#T1, P1, explained_var1 = PCA_nipals(Ost, False)
	cpu_time = time.clock() - c0
	print "time: " + str(cpu_time*1000) + " ms\n"
	#print T1
	#print P1
	#print explained_var1

	print 
	print "nipals c (5x)"
	c0 = time.clock()
	#T2, P2, explained_var2 = PCA_nipals_c(Ost, False)
	#T2, P2, explained_var2 = PCA_nipals_c(Ost, False)
	#T2, P2, explained_var2 = PCA_nipals_c(Ost, False)
	#T2, P2, explained_var2 = PCA_nipals_c(Ost, False)
	#T2, P2, explained_var2 = PCA_nipals_c(Ost, False)
	cpu_time = time.clock() - c0
	print "time: " + str(cpu_time*1000) + " ms\n"
	#print T2
	#print P2
	#print explained_var2
        """
 

 
	#print T1
	#print P1
	#print explained_var1

        if ost:
		print(str(len(list(Cheese.keys()))) + "X PCA of matrix A (cheese data set)")
		print("Scores, loadings and total explained variances are retrieved")
		print()
                (rows, cols) = shape(X[1])

                PCs_ = PCs #min(rows, cols)
                PCs_2 = PCs #min(rows, cols)
                

		print("numpy.linalg.svd PCA (numpy array)")
		c0 = time.clock()
		for i in list(Cheese.keys()):
		    T1, P1, explained_var1 = PCA_svd(X[i], False)
		cpu_time = time.clock() - c0
		print(str(min(rows, cols)) + " PCs, time: " + str(cpu_time*1000) + " ms\n")  

               
		print("py-nipals PCA (numpy matrix)")
		c0 = time.clock()
		for i in list(Cheese.keys()):
		    #pass
		    T2, P2, explained_var2 = PCA_nipals(X[i], False, PCs_, t)
		cpu_time = time.clock() - c0
		print(str(PCs_) + " PCs, time: " + str(cpu_time*1000) + " ms\n")
		#print T2
		#print P2
		#print explained_var2

		print("py-nipals PCA (numpy array)")
		c0 = time.clock()
		for i in list(Cheese.keys()):
		    #pass
		    T3, P3, explained_var3 = PCA_nipals2(X[i], False, PCs_, t)
		cpu_time = time.clock() - c0
		print(str(PCs_) + " PCs, time: " + str(cpu_time*1000) + " ms\n")
		#print T3
		#print P3
		#print explained_var3

		print("c-nipals PCA (numpy array)")
		c0 = time.clock()
		for i in list(Cheese.keys()):
		    T4, P4, explained_var4 = PCA_nipals_c(X[i], False, PCs_2, t)
		cpu_time = time.clock() - c0
		print(str(PCs_2) + " PCs, time: " + str(cpu_time*1000) + " ms\n")
		#print T4
		#print P4
		#print explained_var4


 
 
 
 
 
 
        else:

		print(str(times) + "X PCA of matrix A (" + str(M) + " x " + str(N) + ") of random values")
		print("Scores, loadings and total explained variances are retrieved")
		print()





		#print T1
		#print P1
		#print explained_var1

		print("py nipals PCA (numpy matrix)")
		c0 = time.clock()
		for i in range(0, times):
		    #pass
		    T2, P2, explained_var2 = PCA_nipals(X, False, PCs, t)
		cpu_time = time.clock() - c0
		print(str(PCs) + " PCs, time: " + str(cpu_time*1000) + " ms\n")
		#print T2
		#print P2
		#print explained_var2

		print("py nipals PCA (numpy array)")
		c0 = time.clock()
		for i in range(0, times):
		    #pass
		    T3, P3, explained_var3 = PCA_nipals2(X, False, PCs, t)
		cpu_time = time.clock() - c0
		print(str(PCs) + " PCs, time: " + str(cpu_time*1000) + " ms\n")
		#print T3
		#print P3
		#print explained_var3

		print("c nipals PCA (numpy array)")
		c0 = time.clock()
		for i in range(0, times):
		    T4, P4, explained_var4 = PCA_nipals_c(X, False, PCs, t)
		cpu_time = time.clock() - c0
		print(str(PCs) + " PCs, time: " + str(cpu_time*1000) + " ms\n")
		#print T4
		#print P4
		#print explained_var4

		print("numpy.linalg.svd PCA (numpy array)")
		c0 = time.clock()
		for i in range(0, times):
		    T1, P1, explained_var1 = PCA_svd(X, False)
		cpu_time = time.clock() - c0
		print(str(min(N,M)) + " PCs, time: " + str(cpu_time*1000) + " ms\n")


"""

12X PCA of matrix A (cheese data set)
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
14 PCs, time: 20.0 ms

py-nipals PCA (numpy matrix)
14 PCs, time: 930.0 ms

py-nipals PCA (numpy array)
14 PCs, time: 950.0 ms

c-nipals PCA (numpy array)
14 PCs, time: 10.0 ms


12X PCA of matrix A (cheese data set)
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
14 PCs, time: 20.0 ms

py-nipals PCA (numpy matrix)
4 PCs, time: 290.0 ms

py-nipals PCA (numpy array)
4 PCs, time: 340.0 ms

c-nipals PCA (numpy array)
4 PCs, time: 10.0 ms





1X PCA of matrix A (17 x 336) of random values
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
17 PCs, time: 50.0 ms

py nipals PCA (numpy matrix)
4 PCs, time: 420.0 ms

py nipals PCA (numpy array)
4 PCs, time: 590.0 ms

c nipals PCA (numpy array)
17 PCs, time: 20.0 ms



12X PCA of matrix A (17 x 28) of random values
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
17 PCs, time: 20.0 ms

py nipals PCA (numpy matrix)
4 PCs, time: 530.0 ms

py nipals PCA (numpy array)
4 PCs, time: 600.0 ms

c nipals PCA (numpy array)
17 PCs, time: 10.0 ms



15X PCA of matrix A (40 x 20) of random values
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
20 PCs, time: 40.0 ms

py nipals PCA (numpy matrix)
20 PCs, time: 5260.0 ms

py nipals PCA (numpy array)
20 PCs, time: 5450.0 ms

c nipals PCA (numpy array)
20 PCs, time: 40.0 ms



1X PCA of matrix A (1000 x 200) of random values
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
200 PCs, time: 2420.0 ms

py nipals PCA (numpy matrix)
4 PCs, time: 14710.0 ms

py nipals PCA (numpy array)
4 PCs, time: 6470.0 ms

c nipals PCA (numpy array)
200 PCs, time: 8110.0 ms



1X PCA of matrix A (2000 x 100) of random values
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
100 PCs, time: 8770.0 ms

py nipals PCA (numpy matrix)
4 PCs, time: 15100.0 ms

py nipals PCA (numpy array)
4 PCs, time: 12880.0 ms

c nipals PCA (numpy array)
100 PCs, time: 4700.0 ms



10X PCA of matrix A (1000 x 1000) of random values
Scores, loadings and total explained variances are retrieved

numpy.linalg.svd PCA (numpy array)
1000 PCs, time: 118620.0 ms

py nipals PCA (numpy matrix)
4 PCs, time: 765490.0 ms

py nipals PCA (numpy array)
4 PCs, time: 188840.0 ms

c nipals PCA (numpy array)
4 PCs, time: 18740.0 ms

"""


if __name__ == '__main__':
    speed_test()
    unittest.main()
