from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from all_imports import *
from preprocess import Preprocess


class PCA1(Preprocess) :

    def __init__(self):
        super().__init__()

        print (" **PCA** Object created")
      
      
    def explore_eigenvalues(self, X_train, X_test) :
        """Scaling data before classifier."""

        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        print ("Train feature shape:", X_train.shape)
        print("Train feature shape:", X_test.shape)
        

        mean_vec = np.mean(X_test, axis=0)
        cov_mat = (X_test - mean_vec).T.dot((X_test - mean_vec)) / (X_test.shape[0]-1)
        print('Covariance matrix \n%s' %cov_mat)
        print('NumPy covariance matrix: \n%s' %np.cov(X_test.T))
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        print('Eigenvectors \n%s' %eig_vecs)
        print("---------------------------------")
        print('\nEigenvalues \n%s' %eig_vals)
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

       # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

       # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        print("---------------------------------")
        print('Eigenvalues in descending order:')
        for i in eig_pairs:
             print(i[0])
        # tot = sum(eig_vals)
        # var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
        # print("Explained variance : \n{}".format(var_exp))
      

        return X_train, X_test
    
    def apply_LDA(self, X_train, X_test, y_train):
        lda = LDA()
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        return X_train, X_test,y_train
        
    
    def apply_PCA(self, X_train, X_test):
        pca = PCA(.95)
        pca.fit(X_train)
       
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        print("Number of components:\n{}", pca.n_components_)
        print("pcs train:\n{}", X_train) 
        print("---------------------------------")
        print('Explained variation per principal component:\n {}'.format(pca.explained_variance_ratio_))
        print("---------------------------------")
        print("PCA singular values:\n {}",pca.singular_values_)
        print("---------------------------------")
        print("pcs train:\n{}", X_test) 
        print("---------------------------------")
        # plt.plot(np.arange(1,11),np.cumsum(pca.explained_variance_ratio_));
        # plt.scatter(X_train[:,0], X_train[:,1]);
        # plt.axis('equal')
        # plt.scatter(X_train[:,0], X_train[:,1]);
        # plt.show()
       
        return  X_train, X_test
          
  
    def apply_kernelPCA(self, X_train, X_test, num_comp):
        transformer = KernelPCA(kernel="rbf", n_components=num_comp, n_jobs=-1, random_state=42)
        X_train= transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
        return  X_train, X_test 

    def apply_swissroll(self, X_train, X_test):
        embedding = LocallyLinearEmbedding(n_neighbors=15,n_components=2,method='standard')
        X_train = embedding.fit_transform(X_train)
        X_test = embedding.transform(X_test)
        return X_train, X_test


    def apply_isomapEmbedding(self, X_train, X_test):
           """Returns the embedded points for Isomap."""
           embedding = Isomap(n_components =2, n_jobs=-1)
           X_train = embedding.fit_transform(X_train)
           X_test = embedding.transform(X_test)
           return X_train, X_test
    
    def apply_tsne(self, X_train, X_test):
        pca = PCA(.95)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        X_train = TSNE(random_state=42, perplexity=40, n_iter=2000).fit_transform(X_train)
        return X_train, X_test






# fashion_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)