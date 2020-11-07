import numpy as np
import joblib
from scipy.special import digamma
import sys
from tqdm import trange
from sklearn.metrics import mean_squared_error

def mse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_squared_error(prediction, ground_truth)


def solve_quadratic(A, B, C):
    """
    Solve a quadratic equation in the form Ax^2 + Bx + C
    """
    if abs(A*C) < 1e-10 or abs(A) < 1e-10:
        if -C/B > 1e-10:
            return -C/B
        else:
            return 1e-10
    s1 = (-B + np.sqrt(B**2 - 4*A*C)) / (2 * A)
    s2 = (-B - np.sqrt(B**2 - 4*A*C)) / (2 * A)
    if 0.0 < s1 <= 1.0 and 0.0 < s2 <= 1.0:
        if s1 < s2:
            return s1
        else:
            return s2
    elif 0 < s1 <= 1:
        return s1
    elif 0 < s2 <= 1:
        return s2
    else:
        print(f'A: {A}, B: {B}, C: {C}')
        raise ValueError(f'WARNING: s1 {s1} and s2 {s2} are out of range in solve_quadratic')

class BNPPAPF():
    """
    Initialize a Feature-aware Bayesian Nonparametric Poisson Factorization Recommender System.
    Model by Mario Russo (2020)
    """
    def __init__(self, T, alpha, c, a, b, e_1, f_1, g, export_path=None):
        """
        Initialize a Feature-aware Bayesian Nonparametric Poisson Factorization Recommender System.
        Model by Mario Russo (2020)

        :param T: truncation level T after which we assume the latent features to be distributed according to
                  our prior distribution. Keep in mind that the nonparametric model is meant to allow for generous
                  values of T and obtain a posteriori values of the true dimensionality of the latent vectors by keeping
                  the first K <= T dimensions that explain at least 95% of the forecasted ratings.
                  From an algorithmic point of view, T will be the dimensionality of the latent arrays.
        :type T: int
        :param alpha: hyperparameter on the Beta(1, alpha) and Gamma(alpha, c) distribution that make up the Gamma
                      process used to model the vectors of user latent preferences.
                      Due to the derivation of the CAVI algorithm used, it must be greater than 1.
        :type alpha: int or float
        :param c: rate hyperparameter on the Gamma(alpha, c) distribution that is part of the gamma process that models
                  the vectors of user latent preferences.
        :type c: int or float
        :param a: rate hyperparameter on the Gamma(a, b) distribution that regulates the generative process of each
                  latent item qualities within the latent vector of item qualities.
        :type a: int or float
        :param b: shape hyperparameter on the Gamma(a, b) distribution that regulates the generative process of each
                  latent item qualities within the latent vector of item qualities.
        :type b: int or float
        :param e_1: shape hyperparameter on the Gamma(e_1, e_1/f_1) that regulates the distribution of latent
                    user feature sensitivities.
        :type e_1: int or float
        :param f_1: hyperparameter on the Gamma(e_1, e_1/f_1) that regulates the distribution of latent user feature
                    sensitivities. It represents our prior belief on the expected user feature sensitivity.
        :type g: int or float
        :param g: prior shape hyperparameter on the Gamma(g, g/P_i) prior for item perceived feature values. Note that
                  P_i is the observed feature value and is specific for each item.
        :param export_path: path to export the model to when the train/val error lowers.
                            If val is provided, model is exported when validation error lowers.
                            If val is not provided, model is exported when train error lowers.
                            Path must include export name in .pkl format.
        :type export_path: str or os.path or None, default None.
        """

        self.T = T
        self.alpha = alpha
        self.c = c
        self.a = a
        self.b = b
        self.e_1 = e_1
        self.f_1 = f_1
        self.g = g
        self.export_path = export_path


    def fit(self, epochs, train, features, val=None, seed=123):
        """
        Fit a Bayesian nonparametric Feature-aware Poisson Factorization Model to
        training data.

        :param epochs: number of training epochs.
        :type epochs: int
        :param train: (U X I) array where each row is a user, each column is
                      an item. Used to train the model.
        :type train: numpy.array
        :param features: I-dimensional array of observed item features.
        :type features: numpy.array
        :param val: (U X I) array where each row is a user, each column is
                    an item. Used to validate the model.
        :type val: numpy.array
        :param seed: seed for the random initialization of the variational parameters.
        :type seed: int
        """
        
        # initialize error lists
        self.train_error = []
        self.U, self.I = train.shape

        if val.any():
            self.val_error = []

        # intialize variational parameters to the prior
        self.__initialize_variational_params(features=features, seed=seed)

        self.resume_training(epochs=epochs, train=train, features=features, val=val)


    def resume_training(self, epochs, train, features, val=None):
        """
        Fit a Bayesian nonparametric Feature-aware Poisson Factorization Model to
        training data.

        :param epochs: number of training epochs.
        :type epochs: int
        :param train: (U X I) array where each row is a user, each column is
                      an item. Used to train the model.
        :type train: numpy.array
        :param features: I-dimensional array of observed item features.
        :type features: numpy.array
        :param val: (U X I) array where each row is a user, each column is
                    an item. Used to validate the model.
        :type val: numpy.array
        """
        U, I = train.shape
        nonzero_u, nonzero_i = train.nonzero()
        pbar = trange(epochs, file=sys.stdout, desc = "BNPPAPF")
        for iteration in pbar:
            
            # cache common computations
            expected_s = self.gamma_shp/self.gamma_rte # for cache
            expected_beta = self.lambda_shp/self.lambda_rte
            # logprod (U x T) array of the log stick-breaking process
            logprod = np.array([[np.log(self.tau[u,k]) + np.sum(np.log(1 - self.tau[u, :k])) \
                                for k in range(self.T)] for u in range(U)])


            # for each each u,i for which the rating is > 0:
            for u, i in zip(nonzero_u, nonzero_i):
                # update the variational multinomial parameter
                # first, compute the unnormalized probability of the interaction between sigma_u and delta_i
                self.phi[u, i, 0] = np.exp(digamma(self.mu_shp[u]) - np.log(self.mu_rte)
                        + digamma(self.delta_shp[i]) - np.log(self.delta_rte[i]))
                # then, compute the unnormalized (1 X T) variational probability vector
                self.phi[u, i, 1:] = np.exp(
                                        digamma(self.gamma_shp[u]) - np.log(self.gamma_rte[u])
                                        + logprod[u, :]
                                        + digamma(self.lambda_shp[i, :]) - np.log(self.lambda_rte[i, :])
                                        )
                
                # second, obtain the normalization constant, 
                # made by the finite sum of the above vector (norm1) and the infinite sum (norm2)
                norm2 = (np.exp(
                                digamma(self.gamma_shp[u]) - np.log(self.gamma_rte[u])
                                + digamma(1) - digamma(self.alpha + 1) + np.sum(np.log(1 - self.tau[u, :]))
                                + digamma(self.a) - np.log(self.b)
                               )
                        / (
                            1 - np.exp(digamma(self.alpha) - digamma(self.alpha+1))
                          ) 
                        )
                
                # third, normalize the unnormalized probability vector
                self.phi[u,i] /= np.sum(self.phi[u,i]) + norm2

            
            # for each user, update the feature sensitivity rate and shape, scaling, and stick proportions
            # mu_rte is a scalar, mu_shp is a (U X 1) array
            self.mu_rte = np.sum(self.delta_shp/self.delta_rte) + self.e_1/self.f_1
            self.mu_shp = np.sum(train * self.phi[:, :, 0], axis=1) + self.e_1
            # update gamma_shp and gamma_rte:
            self.gamma_shp = np.sum(train * (1 - self.phi[:, :, 0]), axis=1) + self.alpha
            for u in range(self.U):
                # update scaling
                self.gamma_rte[u] = self.c + np.sum([self.tau[u,k]*np.prod(1-self.tau[u,:k]) \
                                                     * np.sum(self.lambda_shp[k]/self.lambda_rte[k])
                                                     for k in range(self.T)]) \
                                    + self.I * self.a/self.b * np.prod(1 - self.tau[u, :])
               
                # update stick proportions
                for k in range(self.T):
                    #DoubleChecked
                    A = expected_s[u] \
                        * (np.sum([self.tau[u,l]*np.prod(1-self.tau[u,:l])/(1-self.tau[u,k])*np.sum(expected_beta[:, l])
                                   for l in range(k+1, self.T)])
                           - np.prod(1-self.tau[u,:k]) * np.sum(expected_beta[:, k])
                           + self.I * self.a/self.b * np.prod(1-self.tau[u,:])/(1-self.tau[u,k])
                          )       
                    
                    #DoubleChecked
                    C = - np.sum(train[u, :] * self.phi[u, :, k+1])
                    cached_sum = np.sum(train[u, :] * (1 - np.sum(self.phi[u, :, :(k+2)], axis=1)))
                    B = self.alpha - 1 - C - A + cached_sum
                    try:
                        self.tau[u,k] = solve_quadratic(A, B, C)
                    except:
                        raise Exception(f"Error with u,k: {u,k}")
            expected_s = self.gamma_shp/self.gamma_rte


            #for each item, update perceived feature rate and shape, item weight, and popularity parameters
            self.delta_rte = np.sum(self.mu_shp/self.mu_rte) + self.g / features
            self.delta_shp = np.sum(train * self.phi[:, :, 0], axis=0) + self.g
            self.lambda_shp = self.a + np.sum(train.reshape((self.U, self.I ,1)) * self.phi[:, :, 1:], axis=0)
            for i in range(self.I):
                self.lambda_rte[i] = [self.b+np.sum(expected_s * self.tau[:, k] * np.prod(1 - self.tau[:, :k], axis=1))
                                      for k in range(self.T)]
            
            # obtain the latent vectors:
            for k in range(self.T):
                self.theta[:, k] = self.gamma_shp/self.gamma_rte * self.tau[:,k] * np.prod(1 - self.tau[:, :k], axis=1)
            self.beta = self.lambda_shp/self.lambda_rte
            self.pi = self.delta_shp/self.delta_rte
            self.sigma = self.mu_shp/self.mu_rte
            self.prediction = np.dot(np.append(self.sigma.reshape(self.U, -1), self.theta, axis=1),
                                    np.append(self.pi.reshape(self.I, -1), self.beta, axis=1).T)

            self.train_error.append(mse(self.prediction, train))
            if val.any():
                # Note: MSE can be a very misleading measure!
                # Useful only for convergence diagnostic purposes!!
                self.val_error.append(mse(self.prediction, val))
                pbar.set_description(f"BNPPAPF Val MSE(x100): {np.round(self.val_error[-1]*100, 4)} - Progress")
                # export model if export_path is provided
                if self.export_path:
                    joblib.dump(self, self.export_path)
            else:
                pbar.set_description(f"BNPPAPF Train MSE(x100): {np.round(self.train_error[-1]*100, 4)} - Progress")
                if self.export_path:
                    joblib.dump(self, self.export_path)


    def __initialize_variational_params(self, features, seed):
        """
        Randomly initialize the variational parameters according to the prior.
        """
        np.random.seed(seed)
        
        #phi : (U X I X [T+1]) matrix of varPars for the multinomial varDist of z_{u,i;k}
        self.phi = np.zeros(shape=(self.U, self.I, self.T+1))
        
        # tau (U X T) array for VarPars of the delta Vardist of the stick proportions v_{u,k}
        self.tau = np.random.beta(a=1, b=self.alpha, size=(self.U, self.T))
        # gamma_rte, gamma_shp (U X 1) arrays for varpars of vardist of s_u
        self.gamma_rte = np.random.uniform(0.85, 1.15, size=(self.U)) * self.c
        self.gamma_shp = np.random.uniform(0.85, 1.15, size=(self.U)) * self.alpha
        # Mu_shp (U X 1) array for shape VarPars of Vardist of sigma_u
        self.mu_shp = np.random.uniform(low=0.85, high=1.15, size=(self.U, 1))*self.e_1
        # mu_rte is a *single value* for rate varpar of vardist of sigma_u. constant across users.
        self.mu_rte = self.e_1/self.f_1
        
        # lambda_shp, lambda_rte (I X T) arrays of varpars for vardist of beta_{i,k}
        self.lambda_shp = np.random.uniform(0.85, 1.15, size=(self.I, self.T)) * self.a
        self.lambda_rte = np.random.uniform(0.85, 1.15, size=(self.I, self.T)) * self.b
        # Delta_shp, delta_rte (I X 1) arrays for VarPars of Vardist of pi_i
        self.delta_shp = np.random.uniform(low=0.85, high=1.15, size=(self.I, 1))*self.g
        self.delta_rte = self.delta_shp/features.reshape(self.I, 1) #we are not adding any stochasticity in features.
        
        #initialize preferences array
        self.theta = np.zeros((self.U, self.T))
        