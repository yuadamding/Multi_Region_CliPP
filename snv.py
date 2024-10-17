'''
----------------------------------------------------------------------
----------------------------------------------------------------------
This file contains main classes and functions for multi-region CliPP
Authors: Yu Ding
Date: 10/04/2024
Email: yding1995@gmail.com; yding4@mdanderson.org
----------------------------------------------------------------------
-----------------------------------------------------------------------
'''
import numpy as np
import scipy as sci
import itertools
import time
################################################################################
################################################################################
################################################################################
################################################################################

class Timer:
    # This is a class, or decorator, to report the running time of a function you specify.
    def __init__(self, func):
        self.func = func
        
    def __call__(self, *args, **kwds):
        start = time.time()
        ret = self.func(*args, **kwds)
        print(f"Time: {time.time() - start}")
        return ret
        
################################################################################
################################################################################
################################################################################
################################################################################

class SNV:
    # This is a class for a specific SNV, containing all of its info over different samples
    '''
        the observed number of variant reads, r_ij: alt_counts
        SNV-specific copy number at the i-th SNV, b_ij: computed by equation (3) in CliPP paper
        total copy numbers for normal cell, c^N_{ij}: normal_cn, typically set as 2
        total copy numbers for tumor cell, c^T_{ij}: major_cn + minor_cn
        copy number of major allele covering i-th SNV, m_{ij}: major_cn
        total number of reads, n_{ij}: alt_counts + ref_counts
    '''
    
    def __init__(self, 
                 name = None, 
                 num_regions = None,
                 reads = None, 
                 total_reads = None, 
                 rho = None,
                 tumor_cn = None,
                 normal_cn = None,
                 major_cn = None,
                 minor_cn = None,
                 cp = None,
                 prop = None,
                 specific_copy_number = None, 
                 likelihood = None,
                 mapped_cp = None,
                 map_method = "Gaussian",
                 c = None):
        # @param name: mutation name or mutation id
        # @param num_regions: number of samples or regions collected
        # @param reads: the observed number of variant reads, r_ij
        # @param total_reads: total number of reads, n_{ij}
        # @param rho: tumor purity
        # @param tumor_cn: total copy number of tumor cells
        # @param normal_cn: total copy number of normal cells
        # @param major_cn: tumor cell major allele count
        # @param minor_cn: tumor cell minor allele count
        # @param cp: celluar prevalence of SNV, default is 0 over all samples or regions
        # @param prop: expected proportion of this SNV, paramter of Binomial dist
        # @param likelihood: observed log-likelihood function value of this SNV, not it does not contain penalty value
        # @param mapped_cp: using a CDF to convert @param cp, in order to release the box constraint on @param cp
        # @param map_method: the type of CDF to convert @param cp to @param cp, default is "Gaussian"
        # @param c: the coefficience of cp in equation (5) in Xiaoqian's note
        self.name = name
        self.num_regions = num_regions
        self.reads = reads
        self.total_reads = total_reads
        self.rho = rho
        self.tumor_cn = tumor_cn
        self.normal_cn = normal_cn
        self.major_cn = major_cn
        self.minor_cn = minor_cn
        self.cp = cp
        self.prop = prop
        self.specific_copy_number = specific_copy_number
        self.likelihood = likelihood
        self.mapped_cp = mapped_cp
        self.map_method = map_method
        self.c = c

    def initialize_by_matrix(self, mat):
        # This method is to initialize SNV by matrix or dataframe
        self.name = np.unique(mat.mutation)[0]
        self.num_regions = len(np.unique(mat.region))
        self.reads = mat.alt_counts.to_numpy()
        self.total_reads = mat.alt_counts.to_numpy() + mat.ref_counts.to_numpy()
        self.rho = mat.tumour_purity.to_numpy()
        self.tumor_cn = mat.major_cn.to_numpy() + mat.minor_cn.to_numpy()
        self.normal_cn = mat.normal_cn.to_numpy()
        self.major_cn = mat.major_cn.to_numpy()
        self.minor_cn = mat.minor_cn.to_numpy()
        # Set cp equal to 0.5, so that the initial value of mapped cp is 0
        self.cp = np.array([0.5 for i in range(self.num_regions)])
        self.get_specific_copy_number()
        self.get_mapped_cp(method = self.map_method)
        self.c = self.get_c()
        self.prop = self.get_prop()
        
        # Final check
        self.self_check()

    def self_check(self):
        # This method is to test whether initialization is correct.
        pass
    
    def get_specific_copy_number(self):
        # This method is to compute SNV-specific copy number at the i-th SNV, b_ij, according to equation (3) in CliPP paper
        if self.specific_copy_number is not None:
            return self.specific_copy_number
        temp = np.round((self.rho * self.tumor_cn + self.normal_cn * (1 - self.rho)) * self.reads / (self.total_reads * self.rho))
        temp = [temp[i] if temp[i] > 0 else 1 for i in range(self.num_regions)]
        self.specific_copy_number = np.array([np.min([temp[i], self.major_cn[i]]) for i in range(self.num_regions)])
        return self.specific_copy_number
    
    def get_c(self):
        # This method is to compute the coefficience of cp in equation (5) in Xiaoqian's note
        if self.c is not None:
            return self.prop
        temp = (1 - self.rho) * self.normal_cn + self.rho * self.tumor_cn
        self.c = self.specific_copy_number / temp
        return self.c
    
    def get_prop(self):
        # This method is to compute expected proportion, \theta_{ij}, according to equation (1) in CliPP paper
        if self.prop is not None:
            return self.prop
        temp = (1 - self.rho) * self.normal_cn + self.rho * self.tumor_cn
        self.prop = self.cp * self.specific_copy_number / temp
        return self.prop
        
    def get_likelihood(self):
        # This method is to compute the likelihood function value of this specific SNV across all samples or regions.
        # Note the return value does not contain penalty terms.
        # Please refer to equation (2) in CliPP paper
        if self.likelihood is not None:
            return self.likelihood
        temp = self.reads * np.log(self.prop) + (self.total_reads - self.reads) * np.log(1 - self.prop)
        self.likelihood = np.sum(temp)  
        return self.likelihood
    
    def get_mapped_cp(self, method = "Gaussian"):
        # @method: which kind of CDF for this mapping, default is standard gaussian
        # Please refer to equation (7) in CliPP paper
        if self.mapped_cp is not None:
            return self.mapped_cp
        
        if method == "Gaussian" or method == "gaussian":
            self.mapped_cp = [sci.stats.norm.ppf(self.cp[i]) for i in range(self.num_regions)]
            self.mapped_cp = np.array(self.mapped_cp)
            return self.mapped_cp
        
        if method == "Logistic" or method == "logistic":
            self.mapped_cp = [np.log(self.cp[i] / (1 - self.cp[i])) for i in range(self.num_regions)]
            self.mapped_cp = np.array(self.mapped_cp)
            return self.mapped_cp
    
    
    def gradient(self):
        # This method computes the gradient of binomial log-likelihood function, given observations of a single snv    
        # @param snv: a object of class SNV
    
        r_i = self.reads
        c_i = self.c
        cp = self.cp
        n_i = self.total_reads
        f_prime = sci.stats.norm.pdf(self.mapped_cp)
        return r_i * c_i * f_prime / (c_i * cp) - (n_i - r_i) * c_i * f_prime / (1 - c_i * cp)
    
    def set_mapped_cp(self, p):
        # This method updates mapped_cp, given the latest computation results   
        # @param p: a vector of mapped_cp
        self.mapped_cp = p
        self.cp = [sci.stats.norm.cdf(self.mapped_cp[i]) for i in range(self.num_regions)]
        temp = (1 - self.rho) * self.normal_cn + self.rho * self.tumor_cn
        self.prop = self.cp * self.specific_copy_number / temp
        self.likelihood = np.sum(self.reads * np.log(self.prop) + (self.total_reads - self.reads) * np.log(1 - self.prop))
        
    def set_cp(self, method = "Gaussian"):
        # This method should be called only once mapped_cp is updated
        self.cp = [sci.stats.norm.cdf(self.mapped_cp[i]) for i in range(self.num_regions)]

    def get_likelihood_given_p(self, p):
        cp = [sci.stats.norm.cdf(p[i]) for i in range(self.num_regions)]
        temp = (1 - self.rho) * self.normal_cn + self.rho * self.tumor_cn
        prop = cp * self.specific_copy_number / temp
        likelihood = np.sum(self.reads * np.log(prop) + (self.total_reads - self.reads) * np.log(1 - prop))
        return likelihood

################################################################################
################################################################################
################################################################################
################################################################################

class snvs:
    
    def __init__(self, 
                 num_snvs = None,
                 num_regions = None,
                 likelihood = None,
                 snv_lst = None,
                 paris_mapping = None,
                 paris_mapping_inverse = None,
                 p = None,
                 combination = None,
                 v = None,
                 y = None,
                 gamma = 1,
                 omega = None
                 ):
        # @param v: new variable measures differents beween two centroids
        # @param y: lagrangian multipler
        # @param snv_lst: a list contains all snv objects
        # @param num_snvs: the number of snvs
        # @param num_regions: the number of samples
        # @param combination: set that contains all snv pair
        # @param paris_mapping: a mapping from all snv pair to index
        # @param likelihood: the log-likelihood function value of all observations given the current parameters
        # @param omega: the penalty weight vector of the original objective function
        # @param paris_mapping_inverse: the inverse of paris_mapping, mapping from l to (l1, l2)
        # @param gamma: rho in the original documents, the parameter used in the augmented lagrangian method
        
        self.num_snvs = num_snvs
        self.num_regions = num_regions
        self.likelihood = likelihood
        self.snv_lst = snv_lst
        self.paris_mapping = paris_mapping
        self.paris_mapping_inverse = paris_mapping_inverse
        self.p = p
        self.combination = combination
        self.v = v
        self.y = y
        self.gamma = gamma
        self.omega = omega
    
    def initialize_by_list(self, snv_lst):
        self.snv_lst = snv_lst
        self.num_snvs = len(snv_lst)
        self.num_regions = snv_lst[0].num_regions
        self.paris_mapping = self.set_paris_mapping()
        self.p = self.set_p()
        self.v = self.set_v()
        self.y = self.set_y()
        self.likelihood = np.sum([snv_lst[i].likelihood for i in range(self.num_snvs)])
        self.omega = np.ones(len(self.paris_mapping))
        
    def __getitem__(self,
                    subscript):
        return self.snv_lst[subscript]
    
    def get_likelihood(self):
        self.likelihood = np.sum([self.snv_lst[i].likelihood for i in range(self.num_snvs)])
        
    def set_omega(self, omega):
        if isinstance(omega, int):
            self.omega = [omega for i in range(len(self.paris_mapping))]
        
    
    def set_omega_set(self):
        # This method generates a Omega Set that contains all of possible paris of SNVs
        sets = {i for i in range(self.num_snvs)}
        combinations_2 = list(itertools.combinations(sets, 2))
        self.combination = combinations_2
    
    def set_paris_mapping(self):
    # This method gives a dic contains a mapping from paris to index
    # For example, (0, 1) is the pair of SNV 1 and SNV 2
    # This dic will map (0, 1) to 1, so that 1 is the index for variable p, v, and y
    
    # @param n: number of SNV
        if self.combination is None:
            self.set_omega_set()
            
        combinations_2 = self.combination
        dic1 = {}
        dic2 = {}
        index = 0
        for i in range(len(combinations_2)):
            combination = combinations_2[i]
            dic1[combination] = index
            dic2[index] = combination
            index = index + 1
        
        self.paris_mapping = dic1
        self.paris_mapping_inverse = dic2
        return dic1

    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_p(self, p = None):
        # This method generates a vector with length N*M, contains all mapped 
        # celluar prevalence of all snv_s
        if self.p is not None:
            self.p = p
            p = np.reshape(p, (self.num_snvs, self.num_regions))
            for i in range(self.num_snvs):
                self.snv_lst[i].set_mapped_cp(p[i, :])
            return 
            
        p = []
        for i in range(self.num_snvs):
            p.extend(self.snv_lst[i].mapped_cp)
        
        self.p = np.array(p)
        return self.p
    
    def set_p_single_snv(self, p, index):
        start = index * self.num_regions 
        end = (index + 1) * self.num_regions 
        self.p[start:end] = p
        self.snv_lst[index].set_mapped_cp(p)
    
    def set_v(self, v = None, index = None):
        # This method initialize v
        if self.v is not None:
            start = index * self.num_regions
            end = (index + 1) * self.num_regions
            self.v[start : end] = v
            return 
        
        if self.combination is None:
            self.set_omega_set()
            
        v = []
        for i in range(len(self.combination)):
            a, b = self.combination[i]
            tmp = self.snv_lst[a].mapped_cp - self.snv_lst[b].mapped_cp
            v.extend(tmp)
        
        self.v = np.array(v)
        return self.v
    
    def get_v(self, index):
        start = index * self.num_regions
        end = (index + 1) * self.num_regions
        return self.v[start : end]
    
    def set_y(self):
        # This method initialize y
         
        self.y = np.ones(self.num_regions * len(self.combination))
        return self.y
    
    def get_likelihood(self, p):
        p = np.reshape(p, (self.num_snvs, self.num_regions))
        res = np.sum(self.snv_lst[i].get_likelihood_given_p(p[i, :]) for i in range(self.num_snvs))
        return res
    
    def obj_func_p(self):
        # This method generates the objective function for P, in the ADMM iterations.
        # Note the propose of this function is for testing by using existing optimization solver
    
        v_tilde = self.v + self.y / self.gamma
        
        def func(p):
            # This is a function to be set up
        
            # @param p: the vectorized form of P matrix.
            
            ret = -self.get_likelihood(p)
            for i in range(len(self.combination)):
                combination = self.combination[i]
                (v1, v2) = combination
                a_l = a_mat_generator(v1, v2, self.num_snvs, self.num_regions)
                index = self.paris_mapping[combination]
                start = index * self.num_regions
                end = (index + 1) * self.num_regions
                tmp = v_tilde[start : end] - np.matmul(a_l, p)
                ret = ret + 0.5 * self.gamma * np.matmul(tmp.T, tmp)
            return ret
    
        return func
    
    def obj_func_p2(self, index):
        
        v_tilde = self.v + self.y / self.gamma
        def func2(p):
            # @param p: mapped_cp for a snv only, num_regions * 1
            # @param index: the index of this snv
            
            ret = -self.snv_lst[index].get_likelihood_given_p(p)
            for i in range(self.num_snvs):
                if i != index:
                    pair = (index, i) if index < i else (i, index)
                    index_v = self.paris_mapping[pair]
                    start_v = index_v * self.num_regions
                    end_v = (index_v + 1) * self.num_regions
                    start_p = i * self.num_regions
                    end_p = (i + 1) * self.num_regions
                    tmp = v_tilde[start_v : end_v] - p + self.p[start_p: end_p]
                    ret = ret + 0.5 * self.gamma * np.matmul(tmp.T, tmp) * 0.5
            
            return ret
    
        return func2
    
    def obj_func_v(self, v_index):
        # This method generates the objective function for v_l, in the ADMM iterations.
        # Note the propose of this function is for testing by using existing optimization solver
        
        v1, v2 = self.paris_mapping_inverse[v_index]
        start = v_index * self.num_regions
        end = (v_index + 1) * self.num_regions
        a_l = a_mat_generator(v1, v2, self.num_snvs, self.num_regions)
        temp1 = np.matmul(a_l, self.p)
        y_l = self.y[start : end]
        gamma = self.gamma
        omega = self.omega[v_index]
        
        def func(v):
            tmp1 = v - temp1 + y_l / gamma
            return np.matmul(tmp1.T, tmp1) / 2 + omega * np.matmul(v.T, v) / gamma
        
        return func
        
    def update_y(self, y_index):
        v1, v2 = self.paris_mapping_inverse[y_index]
        a_l = a_mat_generator(v1, v2, self.num_snvs, self.num_regions)
        start = y_index * self.num_regions
        end = (y_index + 1) * self.num_regions
        temp1 = np.matmul(a_l, self.p)
        v_l = self.v[start : end]
        y = self.y[start : end] + self.gamma * (v_l - temp1)
        self.y[start : end] = y
        
    def dis_cluster(self):
        
        res = np.zeros((self.num_snvs, self.num_snvs))
        for i in range(len(self.combination)):
            v = self.get_v(i)
            if np.matmul(v.T, v) < 0.01:
                res[self.combination[i]] = 1
                
        return res

################################################################################
################################################################################
################################################################################
################################################################################

def a_mat_generator(v1, v2, n, m):
    # This function generates A_l matrix, please see notes for definition
    # @param v1 & v2, two distinct SNVs
    # @param n: number of SNVs
    # @param m: number of samples
    
    if n < 2 or m < 1:
        raise("The number of SNVs or number of samples are wrong.")
    
    temp1 = np.zeros(n)
    temp2 = np.zeros(n)
    temp1[v1] = 1
    temp2[v2] = 1
    a_mat = np.kron(temp1 - temp2, np.diag(np.ones(m)))
    return a_mat


def a_trans_a_mat_generator(n, m):
    # This function generates A^T_l %*% A_l matrix, please see notes for definition
    # @param n: number of SNVs
    # @param m: number of samples
    sets = {i for i in range(n)}
    combinations_2 = list(itertools.combinations(sets, 2))

    tmp1 = a_mat_generator(0, 0, n, m)
    tmp1 = np.matmul(tmp1.T, tmp1)
    while len(combinations_2) > 0:
        combination = combinations_2.pop()
        (v1, v2) = combination
        tmp2 = a_mat_generator(v1, v2, n, m)
        tmp1 = tmp1 + np.matmul(tmp2.T, tmp2)
    
    return tmp1
    

def a_trans_a_mat_generator_quick(n, m):
    # This function generates A^T_l %*% A_l matrix, please see notes for definition (quick version)
    # @param n: number of SNVs
    # @param m: number of samples
    pass
    
def gradient(snv):
    # This function computes the gradient of binomial log-likelihood function, given observations of a single snv
    # There is a same method in class SNV, but I also put it here seperately for convenience
    
    # @param snv: a object of class SNV
    
    r_i = snv.reads
    c_i = snv.c
    cp = snv.cp
    n_i = snv.total_reads
    f_prime = sci.stats.norm.pdf(snv.mapped_cp)
    return r_i * c_i * f_prime / (c_i * cp) - (n_i - r_i) * c_i * f_prime / (1 - c_i * cp)

def p_vec_generator(snv_s):
    # This functions generates a p vector with dimension mn * 1
    # @param snv_s: a list contains all snv objects
    
    ret = []
    for i in range(len(snv_s)):
        ret.extend(snv_s[i].mapped_cp)
    return np.array(ret)


         

def obj_func_p(v, y, snv_s, n, m, gamma):
    # This function generates the objective function for P, in the ADMM iterations.
    # Note the propose of this function is for testing by using existing optimization solver
    
    # @param v: new variable measures differents beween two centroids. Note stored in vectorized form dim: NM * 1
    # @param y: lagrangian multipler. Note stored in vectorized form dim: NM * 1
    # @param snv_s: a list contains all snv objects
    # @param n: the number of snvs
    # @param m: the number of samples
    # @param gamma: the penalty coefficient
    
    likelihood = np.sum([snv_s[i].likelihood for i in range(len(snv_s))])
    # The following needs to be revised.
    v_tilde = v + y / gamma
    sets = {i for i in range(n)}
    combinations_2 = list(itertools.combinations(sets, 2))
        
    def func(p):
        # This is a function to be set up
        
        # @param p: the vectorized form of P matrix.
        ret = -likelihood
        while len(combinations_2) > 0:
            combination = combinations_2.pop()
            (v1, v2) = combination
            a_l = a_mat_generator(v1, v2, n, m)
            ret = ret + 0.5 * gamma * np.linalg.norm(v_tilde - np.matmul(a_l, p))**2
        return ret
    
    return func

def obj_func_v(p, y):
    # This function generates the objective function for v, in the ADMM iterations.
    # Note the propose of this function is for testing by using existing optimization solver
    
    def func(v):
        pass
    
    pass



def ADMM(snv_s, gamma, num_regions = None):
    n = len(snv_s)
    m = num_regions if num_regions is not None else snv_s[0].num_regions
    pass


################################################################################
################################################################################
################################################################################
################################################################################

