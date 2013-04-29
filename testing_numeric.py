#!/usr/bin/env python

from Numeric import array

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
	   [7, 3, 4, 1]], Float)

# mean_center(X) 
X_centered = array([[-0.83333333,  0.,          0.83333333, -1.33333333],
                    [-1.83333333,  0.,        -2.16666667,  2.66666667],
                    [ 1.16666667,  3.,          0.83333333,  0.66666667],
                    [-0.83333333, -2.,         -2.16666667, -1.33333333],
                    [-1.83333333, -1.,          1.83333333,  0.66666667],
                    [4.16666667,  0.,          0.83333333, -1.33333333]], Float)

# correct for this X, after calculation (using default parameters): 
# T and P are rotated 180 degrees for NIPALS compared to SVD
Scores_nipals = array([[ 0.05315546, -0.54059043, -1.40856561,  0.94209663],
 		       [-3.16804321,  1.35492341,  1.74278325, -0.50700886],
                       [ 1.91518104,  2.6635957,   0.38996132,  0.76502921],
                       [-1.62061336, -2.74118258,  0.86747728,  0.52314455],
                       [-1.34610462,  0.37552823, -2.32841078, -0.89002855],
                       [ 4.16642469, -1.11227433,  0.73675454, -0.83323299]], Float)
 
Loadings_nipals = array([[ 0.82265384,  0.28955294,  0.36488765, -0.32597047],
                         [-0.11433678,  0.72098754,  0.25287236,  0.63494853],
                         [ 0.42906403,  0.14410694, -0.85475226,  0.25403903],
                         [-0.35506592,  0.61283705, -0.26890734, -0.65272336]], Float)





# correct for this X:          
explained_var = array([ 0.51456118,  0.26253703,  0.17232234,  0.05057945], Float)



# amount of decimals to check
accurate = 6   # used where there should be high accuracy
not_so_accurate = 2    

class TestPCA(unittest.TestCase):   
    def test_centering(self):
        print("test centering")
        # if mean center fails, PCA will fail
        X_c = mean_center(X)
        
        self.failUnlessEqual(X_c.shape, X_centered.shape, 'wrong shape')
        
        for i in range(len(X_centered)):
            for j in range(len(X_centered[0])):
                self.failUnlessAlmostEqual(X_c[i,j], X_centered[i,j], accurate,'wrong value in X_c[%i,%i] (svd)' % (i,j))        
           
    def test_nipals(self):
        print("test nipals")
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, e_var = PCA_nipals(X)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        self.failUnlessEqual(e_var.shape, explained_var.shape, 'wrong shape')
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i]' % (i,j))
        
        #print P
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i]' % (i,j))        
        
        #print e_var
        for i in range(len(explained_var)):
            self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+']')


    def test_nipals2(self):
        print("test nipals2")
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
        
        print(E)
        for i in range(len(explained_var)):
            pass
            #self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+']')

  
    def test_nipals_c(self):
        print("test nipals_c")
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, e_var = PCA_nipals_c(X)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        self.failUnlessEqual(e_var.shape, explained_var.shape, 'wrong shape')
        #print T
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i]' % (i,j))
        #print P
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i]' % (i,j))        
        #print e_var
        #print explained_var
        for i in range(len(explained_var)):
            self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+']')
  
  
    def test_nipals_c2(self):
        print("test nipals_c2")
        # using default parameters (should be: standardize=True, PCs=10, threshold=0.0001)
        T, P, E = PCA_nipals_c(X, E_matrices=True)
        self.failUnlessEqual(T.shape, Scores_nipals.shape, 'wrong shape')
        self.failUnlessEqual(P.shape, Loadings_nipals.shape, 'wrong shape')
        #self.failUnlessEqual(E.shape, explained_var.shape, 'wrong shape')
        #print T
        for i in range(len(Scores_nipals)):
            for j in range(len(Scores_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(T[i,j], Scores_nipals[i,j], accurate, 'wrong value in T[%i,%i]' % (i,j))
        #print P
        for i in range(len(Loadings_nipals)):
            for j in range(len(Loadings_nipals[0])):
                #pass
                self.failUnlessAlmostEqual(P[i,j], Loadings_nipals[i,j], accurate, 'wrong value in P[%i,%i]' % (i,j))        
        #print e_var
        #print explained_var
        print(E)
        for i in range(len(explained_var)):
            pass
            #self.failUnlessAlmostEqual(e_var[i], explained_var[i], not_so_accurate, 'wrong value in e_var['+str(i)+']')
    
       
        
        
############## SPEED TESTING ############## 

def speed_test():
        # averaged Cheese data set:
	Ost = array([[ 5.72916667,  3.41666667,  3.175     ,  2.075     ,  1.28333333,
		 2.61666667,  6.22083333,  3.55416667,  2.39583333,  4.67916667,
		 4.32916667,  2.95      ,  2.63333333,  1.99166667,  1.25      ,
		 2.87083333,  4.11666667],
	       [ 6.075     ,  2.74166667,  3.63333333,  2.22916667,  1.225     ,
		 3.38333333,  6.17083333,  2.775     ,  2.14166667,  4.26666667,
		 4.11666667,  3.37083333,  2.67083333,  2.04583333,  1.38333333,
		 3.4       ,  4.0875    ],
	       [ 6.11666667,  3.49166667,  3.52083333,  1.89166667,  1.20833333,
		 2.7125    ,  6.1625    ,  3.45833333,  2.25833333,  4.59166667,
		 4.35      ,  3.14583333,  2.76666667,  1.74166667,  1.35416667,
		 3.02916667,  4.18333333],
	       [ 6.0875    ,  3.18333333,  3.84583333,  1.925     ,  1.09166667,
		 3.025     ,  6.19583333,  3.0625    ,  2.20416667,  4.29583333,
		 4.46666667,  3.42083333,  2.57916667,  1.80833333,  1.18333333,
		 2.83333333,  4.37916667],
	       [ 6.64166667,  2.42916667,  4.39583333,  2.18333333,  1.125     ,
		 4.4375    ,  6.54583333,  2.4625    ,  2.10833333,  4.90833333,
		 4.7375    ,  3.94583333,  2.79166667,  2.2125    ,  1.29166667,
		 4.37083333,  3.99166667],
	       [ 5.95833333,  3.7625    ,  3.25833333,  1.90416667,  1.42083333,
		 2.45833333,  6.00833333,  3.86666667,  2.4875    ,  4.10833333,
		 4.09583333,  2.94583333,  2.37083333,  1.81666667,  1.39583333,
		 2.46666667,  4.4375    ],
	       [ 6.4375    ,  2.8375    ,  3.94583333,  2.225     ,  1.575     ,
		 3.49166667,  6.725     ,  2.52083333,  2.20833333,  4.3375    ,
		 4.85833333,  4.05416667,  3.0875    ,  2.35833333,  1.7       ,
		 4.1875    ,  3.8875    ],
	       [ 6.625     ,  2.85833333,  4.25      ,  2.10833333,  1.19583333,
		 3.7875    ,  6.55833333,  2.7       ,  2.21666667,  3.99166667,
		 4.82916667,  3.975     ,  2.85      ,  2.25833333,  1.54583333,
		 4.23333333,  4.01666667],
	       [ 5.99583333,  3.27083333,  3.2625    ,  2.1       ,  1.125     ,
		 2.33333333,  6.23333333,  3.0375    ,  2.37916667,  4.43333333,
		 4.66666667,  3.5625    ,  2.75833333,  2.075     ,  1.45      ,
		 3.125     ,  3.88333333],
	       [ 5.82916667,  3.35416667,  3.21666667,  1.525     ,  1.12916667,
		 1.475     ,  6.10833333,  3.35833333,  2.25833333,  4.2625    ,
		 4.15833333,  3.3375    ,  2.40833333,  1.6       ,  1.25833333,
		 2.35833333,  4.05      ],
	       [ 6.05833333,  3.0375    ,  3.8       ,  1.975     ,  1.15      ,
		 2.95      ,  6.35      ,  2.47916667,  2.22083333,  4.24166667,
		 4.55833333,  3.89583333,  2.8       ,  2.34166667,  1.54166667,
		 3.96666667,  4.12916667],
	       [ 6.33333333,  2.31666667,  4.125     ,  2.52916667,  1.45      ,
		 4.35416667,  6.75      ,  1.92916667,  2.22916667,  4.32916667,
		 4.75      ,  4.15416667,  2.9625    ,  2.64166667,  1.82916667,
		 5.06666667,  4.08333333],
	       [ 5.825     ,  4.82916667,  1.49583333,  1.07083333,  1.        ,
		 1.09583333,  6.1       ,  4.825     ,  2.45833333,  4.65      ,
		 3.99166667,  1.7875    ,  2.22083333,  1.22083333,  1.1375    ,
		 1.24583333,  4.09166667],
	       [ 5.65      ,  4.6625    ,  1.92916667,  1.025     ,  1.        ,
		 1.075     ,  6.025     ,  4.60833333,  2.175     ,  4.97083333,
		 4.12083333,  1.79583333,  2.29166667,  1.        ,  1.025     ,
		 1.025     ,  4.21666667]])        



	X = array([[2, 3, 4, 1],
		   [1, 3, 1, 5],
		   [4, 6, 4, 3],
		   [2, 1, 1, 1],
		   [1, 2, 5, 3],
		   [7, 3, 4, 1]])



if __name__ == '__main__':
    unittest.main()