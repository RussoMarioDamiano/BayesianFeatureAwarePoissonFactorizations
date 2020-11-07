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


class PAHPF():
    """
    Initialize a Feature-aware Hierarchical Poisson Factorization Recommender System.
    Model by Mario Russo (2020)
    """
    def __init__(self, K, a_1, b_1, a, c_1, d_1, c, e_1, f_1, g, export_path=None):
        """
        Initialize a Feature-aware Hierarchical Poisson Factorization Recommender System.
        Model by Mario Russo (2020)

        :param K: dimensionality of the latent preferences and qualities vectors.
        :type K: int
        :param a_1: prior shape hyperparameters on the Gamma(a_1, a_1/b_1) prior for
                    user activity value.
        :type a_1: int or float
        :param b_1: prior expected value on the Gamma(a_1, a_1/b_1) prior for
                    user activity value.
        :type b_1: int or float
        :param a: shape hyperparameter for the Gamma(a, user_activity) prior
                  for the elements of user u's preference vector.
        :type a: int or float
        :param c_1: prior shape hyperparameter on the Gamma(c_1, c_1/d_1) prior for
                    item popularity value.
        :type c_1: int or float
        :param d_1: prior expected value on the Gamma(c_1, c_1/d_1) prior for
                    item popularity value.
        :type d_1: int or float
        :param c: shape hyperparameter for the Gamma(c, item_popularity) prior
                  for the elements of item i's qualities vector.
        :type c: int or float
        :param e_1: prior shape hyperparameter on the Gamma(e_1, e_1/f_1) prior for user feature sensitivities.
        :type e_1: int or float
        :param f_1: prior expected value on the Gamma(e_1, e_1/f_1) prior for user feature sensitivities.
        :type f_1: int or float
        :param g: prior shape hyperparameter on the Gamma(g, g/P_i) prior for item perceived feature values. Note that
                  P_i is the observed feature value and is specific for each item.
        :param export_path: path to export the model to when the train/val error lowers.
                            If val is provided, model is exported when validation error lowers.
                            If val is not provided, model is exported when train error lowers.
                            Path must include export name in .pkl format.
        :type export_path: str or os.path or None, default None.
        """
        self.K = K
        self.a_1 = a_1
        self.b_1 = b_1
        self.a = a
        self.c_1 = c_1
        self.d_1 = d_1
        self.c = c
        self.e_1 = e_1
        self.f_1 = f_1
        self.g = g
        self.export_path = export_path


    def fit(self, epochs, train, features, val=None, seed=123):
        """
        Fit a Feature-aware Hierarchical Poisson Factorization Model to
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
        self.__initialize_variational_params(features, seed=seed)

        self.resume_training(epochs=epochs, train=train, features=features, val=val)


    def resume_training(self, epochs, train, features, val=None):
        """
        Resume PoissonFactorizations training for additional epochs.

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
        nonzero_u, nonzero_i = train.nonzero()
        pbar = trange(epochs, file=sys.stdout, desc = "HPF")
        for iteration in pbar:
            # for each each u,i for which the rating is > 0:
            for u, i in zip(nonzero_u, nonzero_i):

                # update the variational multinomial parameter
                self.phi[u, i, :(self.K-1)] = np.exp(digamma(self.gamma_shp[u]) - np.log(self.gamma_rte[u])
                        + digamma(self.lambda_shp[i]) - np.log(self.lambda_rte[i]))
                #print(digamma(self.mu_shp[u]).shape, self.delta_shp[i].shape, self.delta_rte[i].shape)
                self.phi[u, i, (self.K-1)] = np.exp(digamma(self.mu_shp[u]) - np.log(self.mu_rte)
                        + digamma(self.delta_shp[i]) - np.log(self.delta_rte[i]))
                self.phi[u,i] /= np.sum(self.phi[u,i])

            #for each user, update the user weight and activity parameters
            self.gamma_rte = ((self.kappa_shp/self.kappa_rte).reshape(self.U, -1) +
                              np.sum(self.lambda_shp/self.lambda_rte, axis=0))
            self.gamma_shp = self.a + np.sum(train.reshape(self.U, self.I, -1) * self.phi[:, :, :(self.K-1)], axis=1)
            # update feature sensitivity rate and shape - rte is a scalar, shp is a (U X 1) array
            self.mu_rte = np.sum(self.delta_shp/self.delta_rte) + self.e_1/self.f_1
            self.mu_shp = np.sum(train * self.phi[:, :, (self.K-1)], axis=1) + self.e_1
            # update kappa_rte after gamma_shp and gamma_rte since it depends on them
            self.kappa_rte = (self.a_1/self.b_1) + (self.gamma_shp/self.gamma_rte).sum(axis=1)

            #for each item, update the item weight and popularity parameters
            self.lambda_rte = ((self.tau_shp/self.tau_rte).reshape(self.I, -1) +
                               np.sum(self.gamma_shp/self.gamma_rte, axis=0))
            self.lambda_shp = self.c + np.sum(train.reshape(self.U, self.I, -1) * self.phi[:, :, :(self.K-1)], axis=0)
            #update perceived feature rate and shape - both rte and shp are (I X 1) arrays
            self.delta_rte = np.sum(self.mu_shp/self.mu_rte) + self.g / features
            self.delta_shp = np.sum(train * self.phi[:, :, (self.K-1)], axis=0) + self.g
            # update tau_rte after lambda_shp and lambda_rte since it depends on them
            self.tau_rte = (self.c_1/self.d_1) + (self.lambda_shp/self.lambda_rte).sum(axis=1)

            # obtain the latent vectors:
            self.theta = self.gamma_shp/self.gamma_rte
            self.beta = self.lambda_shp/self.lambda_rte
            self.pi = self.delta_shp/self.delta_rte
            self.sigma = self.mu_shp/self.mu_rte
            self.prediction = np.dot(np.append(self.theta, self.sigma.reshape(self.U, -1), axis=1),
                                    np.append(self.beta, self.pi.reshape(self.I, -1), axis=1).T)

            self.train_error.append(mse(self.prediction, train))
            if val.any():
                # Note: MSE is a very misleading measure!
                # Useful only for convergence diagnostic purposes!!
                self.val_error.append(mse(self.prediction, val))
                pbar.set_description("PoissonFactorizations Val MSE(x100): "
                                     f"{np.round(self.val_error[-1]*100, 4)} - Progress")
                # export model if export_path is provided
                if self.export_path:
                    joblib.dump(self, self.export_path)
            else:
                pbar.set_description("PoissonFactorizations Train MSE(x100): "
                                     f"{np.round(self.train_error[-1]*100, 4)} - Progress")
                if self.export_path:
                    joblib.dump(self, self.export_path)


    def __initialize_variational_params(self, features, seed):
        """
        Randomly initialize the variational parameters according to the prior.
        """
        np.random.seed(seed)

        # immutable parameters
        self.kappa_shp = (self.K - 1)*self.a + self.a_1
        self.tau_shp = (self.K - 1)*self.c + self.c_1

        #phi : (U X I X K) matrix of varPars for the multinomial varDist of z_{u,i;k}, with z_{u,i;K}=sigma_u*pi_i
        self.phi = np.zeros(shape=(self.U, self.I, self.K))

        # Gamma_shp, Gamma_rte (U X K-1) arrays for VarPars of Vardist of theta_{u,k}
        self.gamma_shp = np.random.uniform(low=0.85, high=1.15, size=(self.U, self.K-1))*self.a
        self.gamma_rte = np.random.gamma(shape=self.a_1, scale=self.b_1/self.a_1, size=(self.U, self.K-1))
        # Mu_shp (U X 1) array for shape VarPars of Vardist of sigma_u
        self.mu_shp = np.random.uniform(low=0.85, high=1.15, size=(self.U, 1))*self.e_1
        # mu_rte is a *single value* for rate varpar of vardist of sigma_u. constant across users.
        self.mu_rte = self.e_1/self.f_1
        # kappa_rte (U X 1) array for varpars of vardist of xi_u
        self.kappa_rte = np.random.uniform(low=0.85, high=1.15, size=(self.U, 1))*self.a_1/self.b_1

        # lambda_shp, lambda_rte (I X K-1) arrays of varpars for vardist of beta_{i,k}
        self.lambda_shp = np.random.uniform(low=0.85, high=1.15, size=(self.I, self.K-1))*self.c
        self.lambda_rte = np.random.gamma(shape=self.c_1, scale=self.d_1/self.c_1, size=(self.I, self.K-1))
        # Delta_shp, delta_rte (I X 1) arrays for VarPars of Vardist of pi_i
        self.delta_shp = np.random.uniform(low=0.85, high=1.15, size=(self.I, 1))*self.g
        self.delta_rte = self.delta_shp / features.reshape(self.I, 1) #we are  not adding any stochasticity in features.
        #tau_rte, (I X 1) array of varpars for vardist of eta_i
        self.tau_rte = np.random.uniform(low=0.85, high=1.15, size=(self.I, 1))*self.c_1/self.d_1

