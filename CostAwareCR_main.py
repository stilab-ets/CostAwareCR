#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import os
import copy
import pickle
import time 
import json

import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.algorithms.soo.nonconvex.ga import GA
from sklearn.ensemble import RandomForestClassifier
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.age import AGEMOEA


from pymoo.factory import get_problem
from pymoo.factory import get_sampling, get_crossover, get_mutation,get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from sklearn.ensemble import RandomForestRegressor
from imblearn.under_sampling import RandomUnderSampler
from pymoo.core.problem import Problem



from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import mstats
from rulefit import RuleFit
from collections import OrderedDict
from pymoo.core.problem import StarmapParallelization

import multiprocessing
from multiprocessing import freeze_support
from multiprocessing.pool import ThreadPool



from lightgbm import LGBMClassifier, LGBMRegressor
import seaborn as sns
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
sys.path.append('../utils')
from utils import * 


#global variables 
#Edit this to specify the data path, the considered project. 

PROJECT = 'Eclipse'
DATA_PATH = '../../data'
RESULTS_PATH = f'../../results/{PROJECT}'
CROSS_VALIDATION_DATA_PATH = '../../data/longitudinal_cross_validation'
EXP_ID = 'default'
OUTCOME = 'status'
COST_FUNCTION ='churn_cost_predictions'
FEATURES = ['author_experience','author_merge_ratio', 'author_changes_per_week',
       'author_merge_ratio_in_project', 'total_change_num',
       'author_review_num', 'description_length', 'is_documentation',
       'is_bug_fixing', 'is_feature', 'project_changes_per_week',
       'project_merge_ratio', 'changes_per_author', 'num_of_reviewers',
       'num_of_bot_reviewers', 'avg_reviewer_experience',
       'avg_reviewer_review_count', 'lines_added', 'lines_deleted',
       'files_added', 'files_deleted', 'files_modified', 'num_of_directory',
       'modify_entropy', 'subsystem_num'
            #, 'text_prob_ngram'
           ]
NUMERICAL_FEATURES = [
    'author_experience','author_merge_ratio', 'author_changes_per_week',
       'author_merge_ratio_in_project', 'total_change_num',
       'author_review_num', 'description_length', 'project_changes_per_week',
       'project_merge_ratio', 'changes_per_author', 'num_of_reviewers',
       'num_of_bot_reviewers', 'avg_reviewer_experience',
       'avg_reviewer_review_count', 'lines_added', 'lines_deleted',
       'files_added', 'files_deleted', 'files_modified', 'num_of_directory',
       'modify_entropy', 'subsystem_num'
       # , 'text_prob_ngram'
]
#cross_validation_setup
FOLDS = 10 




#helper functions
def GET_DATA(): 
    return pd.read_csv(os.path.join(DATA_PATH,PROJECT + '.csv'))

def set_global_vars(org) : 
    global PROJECT,RESULTS_PATH 
    PROJECT = org
    RESULTS_PATH = RESULTS_PATH = f'../../results/{PROJECT}'

def evaluate_predictions(predictions,trues,costs,k=0.2) :
       
    density = trues*1.0/costs 
    accs = []
    popts = []
    for i  in range(predictions.shape[1]) : 
        predict_y = np.array(predictions[:,i])
        parameter = np.hstack((np.array([density]).T, np.array([costs]).T, np.array([trues]).T, np.array([predict_y]).T))
        acc, popt = acc_popt(parameter,k=k)
        accs.append(acc)
        popts.append(popt)
        #new_row = {'model':model_name,'model_index':i, 'acc' : acc,'popt' : popt}
        #results.append(new_row)
    return accs, popts

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize(x) : 
    return x/np.sum(x)
     
def get_LOC_cost(df) :
    return np.array(df['lines_added'] + df['lines_deleted'])

def get_sota_model():
    best_n_estimators = 500
    best_learning_rate = 0.01
    seed = 2021
    return LGBMClassifier(class_weight='balanced', n_estimators=best_n_estimators, learning_rate=best_learning_rate,
                          subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))


def prepare_data(df) : 
    total_df = df.copy()
    total_df = total_df.sort_values(by=['change_id'])
    total_df = total_df[(total_df['lines_added'] + total_df['lines_deleted'] > 0) ]
    total_df[OUTCOME] =  total_df[OUTCOME].astype(int)
    _, metadata = preprocess(total_df,preprocessing_metadata = {
        'scaling' : {
            'scaler' : StandardScaler(),
            'features' : NUMERICAL_FEATURES
        }
    })
    return total_df, metadata

def get_worst_optimal_area(data):
    """
    :param data: density_effort_defect_predictDensity
    :return: worst area, optimal area
    """
    total_effort = np.sum(data[:, 1])
    total_defect = np.sum(data[:, 2])
    # calculate actual worst area
    data = data[data[:, 0].argsort()]

    point_x = []
    point_y = []
    x_current = 0
    y_current = 0
    point_x.append(x_current)
    point_y.append(y_current)
    for i in range(data.shape[0]):
        x_current = x_current + data[i, 1]
        y_current = y_current + data[i, 2]
        point_x.append(x_current)
        point_y.append(y_current)
    point_x = np.array(point_x)
    point_y = np.array(point_y)
    point_x = point_x / total_effort
    point_y = point_y / total_defect
    worst_area = np.trapz(point_y, point_x)
    ######
    #plt.plot(point_x, point_y, color="black", linewidth=2.5, linestyle="-", label="worst model")

    # calculate actual optimal area
    point_x = []
    point_y = []
    x_current = 0
    y_current = 0
    point_x.append(x_current)
    point_y.append(y_current)
    i = data.shape[0] - 1
    while i >= 0:
        x_current = x_current + data[i, 1]
        y_current = y_current + data[i, 2]
        point_x.append(x_current)
        point_y.append(y_current)
        i = i - 1
    point_x = np.array(point_x)
    point_y = np.array(point_y)
    point_x = point_x / total_effort
    point_y = point_y / total_defect
    optimal_area = np.trapz(point_y, point_x)

    ####
    #plt.plot(point_x, point_y, color="red", linewidth=2.5, linestyle="-", label="optimal model")

    return worst_area, optimal_area

def acc_popt(data,k = 0.2):
    """
    this function is used to compute the ACC and Popt scores 
    :param data: density_effort_defect_predictDensity
    :return: ACC, Popt
    """
    worst_area, optimal_area = get_worst_optimal_area(data)
    total_effort = np.sum(data[:, 1])
    total_defect = np.sum(data[:, 2])
    # calculate predicted area
    acc_mark = False
    acc = 0
    threshold = k*total_effort
    data = data[data[:, 3].argsort()]
    point_x = []
    point_y = []
    x_current = 0
    y_current = 0
    point_x.append(x_current)
    point_y.append(y_current)
    i = data.shape[0] - 1
    while i >= 0:
        if (acc_mark is False and x_current > threshold) :
            acc = y_current/total_defect
            acc_mark = True
        x_current = x_current + data[i, 1]
        y_current = y_current + data[i, 2]
        point_x.append(x_current)
        point_y.append(y_current)
        i = i - 1
    point_x = np.array(point_x)
    point_y = np.array(point_y)
    point_x = point_x / total_effort
    point_y = point_y / total_defect
    predicted_area = np.trapz(point_y, point_x)
    popt = 1 - (optimal_area - predicted_area)/(optimal_area - worst_area)

    #plt.xlabel("code churn (%)", size='14')  
    #plt.ylabel("defect-inducing changes (%)", size='14')  #


    #plt.plot(point_x, point_y, color="blue", linewidth=2.5, linestyle="-", label="prediction model")
    #plt.legend(loc='upper left')
    #plt.show()

    return acc, popt

def define_algorithm(algorithm_name,pop_size=400,n_objeectives = 2,
                    crossover_op = SBX( prob=0.5, eta=15),
                    mutation_op = PolynomialMutation(eta=20,prob = 0.1),
                    ref_points = np.array([[0,0]])
                    ): 
    
    if algorithm_name == 'AGEMOEA' : 
        print('running: AGEMOEA')
        algorithm = AGEMOEA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=crossover_op,
            mutation=mutation_op,
        )
    if algorithm_name == 'NSGA2':
        print('running: NSGA2')
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=crossover_op,
            mutation=mutation_op,
        )
    
    if algorithm_name == 'RNSGA2' : 
        print('running: RNSGA2')
        ref_points = ref_points
        print(ref_points)
        algorithm = RNSGA2(
            pop_size=pop_size,
            ref_points= ref_points,
            sampling=FloatRandomSampling(),
            crossover=crossover_op,
            mutation=mutation_op,
        )
    if algorithm_name == 'RNSGA3' : 
        print('running: RNSGA3')
        print(pop_size)
        ref_points = ref_points
        print(ref_points)
        algorithm = RNSGA3(
            #pop_size=pop_size,
            ref_points= ref_points,
            pop_per_ref_point=pop_size,
            sampling=FloatRandomSampling(),
            crossover=crossover_op,
            mutation=mutation_op,
        )
   
    if algorithm_name == 'NSGA3' : 
        print('Running NSGA3')
        ref_dirs = get_reference_directions("energy", n_objeectives, pop_size)
        algorithm = NSGA3(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=crossover_op,
            mutation=mutation_op,
            ref_dirs = ref_dirs
        )
    
    if algorithm_name == 'UNSGA3' : 
        print('Running NSGA3')
        ref_dirs = get_reference_directions("energy", n_objeectives, pop_size)
        algorithm = UNSGA3(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=crossover_op,
            mutation=mutation_op,
            ref_dirs = ref_dirs
        )
    if algorithm_name == 'GA' : 
        algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=crossover_op,
            mutation=mutation_op,
            eliminate_duplicates=True)
    return algorithm
                           
def get_sota_model_EA():
    best_n_estimators = 500
    best_learning_rate = 0.01
    seed = 2021
    return LGBMRegressor( n_estimators=best_n_estimators, learning_rate=best_learning_rate,
                          subsample=0.9, subsample_freq=1, random_state=np.random.randint(seed))

def scale_data(data,features = NUMERICAL_FEATURES, scaler =  StandardScaler()) :
    result = copy.deepcopy(data)
    data_scaler = copy.deepcopy(scaler)
    try: 
        check_is_fitted(data_scaler) 
        print('transform')
        result[features] = data_scaler.transform(result[features])
        
    except NotFittedError: 
        print('fit transform')
        result[features] = data_scaler.fit_transform(result[features]) 
    metadata = {
        'scaler' : data_scaler,
        'features' : features
    }
    return result, metadata

def preprocess(data,preprocessing_metadata) : 
    result = data.copy()
    all_metadata = {}
    #step 1: scaling 
    result, scaling_metadata = scale_data(result,preprocessing_metadata['scaling']['features'],
                                          scaler = preprocessing_metadata['scaling']['scaler'])
    all_metadata['scaling'] = scaling_metadata 
    return result, all_metadata

def make_ref_point(ys,costs,normalize_churn=True ) : 
    alpha = 0.5
    print('mean ys ref point:', np.mean(ys))
    minimum_required_costs = np.sum(ys*costs)
    ref_point = np.array([0.0,0.0,minimum_required_costs])
    
    if normalize_churn : 
        minimum_required_costs = minimum_required_costs*1.0 / np.sum(costs)
    ref_point = np.array([0, alpha*1 + (1-alpha)*minimum_required_costs])
    print('Ref point:', ref_point)
    return ref_point #np.array([0,0]
    
def get_best_known_mogp() : 
    algorithm = NSGA2(
    pop_size=400,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.5, eta=15),
    mutation=PolynomialMutation(eta=20,prob = 0.1))
    return algorithm 

def compute_predictions_probabilities(X,weights) : 
    ready_X = np.ones((X.shape[0],X.shape[1] + 1 ))
    ready_X[:,1:] = X.copy()
    weighted_sum = np.dot(ready_X,weights.T)
    exp_weighted_sum = np.exp(-1*weighted_sum)
    probabilities = 1.0/(exp_weighted_sum + 1.0)
    return probabilities

def predict(probabilities,threshold): 
    predictions = probabilities > threshold 
    return predictions.astype(int)

def recall(y_true,predictions): 
        return recall_score(y_true,predictions)
    
def cost_predictions(costs,predictions,normalized  = False):
    all_costs =  np.sum(predictions*costs)
    if normalized : 
        all_costs = all_costs/np.sum(costs)
    return all_costs
    
def cost_probabilities(costs,probabilities): 
    return np.sum(probabilities*costs)

def benefit(y_true,predictions): 
    return np.sum(predictions*y_true)

def AUC(y_true,probabilities): 
    return roc_auc_score(y_true, probabilities)

def MCC(y_true,predictions): 
    return matthews_corrcoef(y_true, predictions)

def F1(y_true,predictions): 
    return f1_score(y_true, predictions)

def Gmean(y_true,prediction): 
    return geometric_mean_score(y_true, prediction)

def misclassification_cost(y_true,predictions,alpha = 10): 
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return (fp + alpha*fn)/len(predictions)

def make_plot(title,sota_values,moea_values): 
    plot = Scatter(title = title)
    plot.add(sota_values, color="red")
    plot.add(moea_values,color = "blue")
    return plot
    
class LearnLrWeights(Problem): 
       """
       This class defines the optimization problem that learns the weights of a logistic regression.
       The class inherits from the Abstract problem class from Pymoo. 
       """

    def __init__(self,train_data_features,y_true,costs,
                 objectives = ['recall','churn_cost_predictions'],
                 lb = -1000,ub=1000,prediction_threshold = 0.5,**kwargs): 
        
        self.train_data_features_df = train_data_features
        nb_variables = len(train_data_features.columns) + 1
        temporary = self.train_data_features_df.to_numpy() 
        self.y_true = y_true
        self.costs = costs
        self.train_data_features_np = np.ones((len(self.train_data_features_df),nb_variables ))
        self.train_data_features_np[:,1:] = temporary
        xl = np.array([lb]*nb_variables)
        xu = np.array([ub]*nb_variables)
        self.prediction_threshold = prediction_threshold
        self.objectives = objectives
        super().__init__(n_var=nb_variables, n_obj=len(objectives), n_constr=0, xl=xl, xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
       """
       This method is inherited from the Problem class and should be implemented. 
       this method is parallelized. 
       """
        pool  = ThreadPool(18)
        F = pool.map(self.evaluate_one_sol, x)
        out['F'] = np.array(F)
    def evaluate_one_sol(self,x):
       """
       This function is used to evaluate a given solution. 
       Given the weights (x), the function computes the prediction of an LR model.
       Then, evaluate the performance of the LR model based on the objective functions.
       """
        out = []
        weighted_sum = np.dot(self.train_data_features_np,x)
        exp_weighted_sum = np.exp(-1*weighted_sum)
        probabilities = 1.0/(1.0 + exp_weighted_sum)
        #compute the prediction of the LR model
        predictions = probabilities > self.prediction_threshold 
        predictions = predictions.astype(int)
        for objective in self.objectives: 
            if objective == "recall": 
                out.append(1-self.recall(predictions))
            
            if objective == "TNR": 
                out.append(1-self.TNR(predictions))

            if objective == "churn_cost_predictions":
                out.append(self.churn_cost_predictions(predictions))
            

            if objective == "normalized_churn_cost_predictions":
                out.append(self.churn_cost_predictions(predictions,normalized=True))
                
            if objective == "churn_cost_probabilities":
                out.append(self.churn_cost_probabilities(probabilities))
                
            if objective == "benefit":
                out.append(sum(self.y_true)-self.benefit(predictions)) 
            
            if objective == "AUC":
                #print('TIC')
                out.append(1 - self.AUC(probabilities)) 
            
            if objective == "MCC":
                out.append(-1*self.MCC(predictions)) 
            
            if objective == "F1":
                out.append(-1*self.F1(predictions)) 
            
            if objective == "Gmean":
                out.append(-1*self.Gmean(predictions)) 
            
            if objective == "misclassification_cost": 
                out.append(self.misclassification_cost(predictions)) 
            
            if objective == 'popt' : 
                acc,popt = evaluate_predictions(np.array([probabilities]).T,self.y_true,self.costs,k=0.2) 
                out.append(1-popt[0])

            if objective == 'average_churn_recall' : 
                #print('TOC')
                alpha = 0.5 
                val = alpha*(1-self.recall(predictions)) + (1 - alpha)*self.churn_cost_predictions(predictions,normalized=True)
                out.append(val)

            if objective == "average_AUC_and_recall_churn" : 
                alpha = 0.5
                recall_churn_val = 0.5*(1-self.recall(predictions)) + 0.5*self.churn_cost_predictions(predictions,normalized=True)
                val = alpha*(1 - self.AUC(probabilities)) + (1-alpha)*recall_churn_val
                out.append(val)
            
            if objective == "average_AUC_recall_churn" : 
                val = ((1 - self.AUC(probabilities)) + (1-self.recall(predictions)) + self.churn_cost_predictions(predictions,normalized=True))/3
                out.append(val)
        return out 

    #the following functions are helper functions for 
    def recall(self,predictions): 
        return recall_score(self.y_true,predictions)
    
    def churn_cost_predictions(self,predictions,normalized = False): 
        return cost_predictions(self.costs,predictions,normalized)
    
    def churn_cost_probabilities(self,probabilities): 
        return np.sum(probabilities*self.costs)
    
    def benefit(self,predictions): 
        return np.sum(predictions*self.y_true)
    
    def AUC(self,probabilities): 
        return roc_auc_score(self.y_true, probabilities)
    
    def MCC(self,predictions): 
        return matthews_corrcoef(self.y_true, predictions)
    
    def F1(self,predictions): 
        return f1_score(self.y_true, predictions)
    
    def Gmean(self,prediction): 
        return geometric_mean_score(self.y_true, prediction)
    
    def misclassification_cost(self,predictions,alpha = 10): 
        tn, fp, fn, tp = confusion_matrix(self.y_true, predictions).ravel()
        return (fp + alpha*fn)/len(predictions)  
    def TNR(self, predictions): 
        return recall_score(self.y_true,predictions,pos_label=0)
           

class MOGA_LR_warapper : 
    def __init__(
                 self,
                 params = {},
                 objectives = ['recall','churn_cost_predictions'],
                 ) : 
        
        #definition variables 
        self.X_train = None
        self.y_train = None 
        self.cost_train = None 
        self.default_params = {
            'GA_algorithm' : 'NSGA2',
            "n_gen" : 200,
            'population_size': 400,
            'crossover_op' : SBX( prob=0.5, eta=15),
            'mutation_op' : PolynomialMutation(eta=20,prob = 1/(len(FEATURES) + 10) ),
            'ref_points' : np.array([0.0,0.0]),
            'use_rulefit': True,
            'rulefit_base_model': RandomForestClassifier(n_estimators = 500, max_depth=5, class_weight='balanced', n_jobs=-1),
            'winsorize' : False,
            'winsorize_limit' :0.025, 
            'top_rules_count': 10,
            'add_sota_prob' : False,
            'SOTA_model' : get_sota_model()
        }
        self.actual_params = copy.deepcopy(self.default_params)
        self.set_params(params)
        #self.prediction_cost_function = prediction_cost_function
        #self.preprocessing_function = preprocessing_function
        self.objectives = objectives
        #state variables 
        self.is_fit = False
        
        self.opt_problem = None
        self.learned_weights = None
        self.weights_objectives = None 
        self.train_indicies=None 
        self.validation_indicies = None
        self.X_val = None 
        self.y_val = None  
        self.best_model_idx= None
        
    def fit(self,X,y,costs,train_indicies = None, validation_indicies = None
            ) :

        self.train_indicies=None 
        self.validation_indicies = None 

        self.X_train = X
        self.y_train = y 
        self.train_costs = costs
        if not (train_indicies is None) : 
            
            self.train_indicies = train_indicies
            self.validation_indicies = validation_indicies
            self.X_train = self.X_train.iloc[self.train_indicies]
            self.y_train = self.y_train.iloc[self.train_indicies]
            self.train_costs = costs[self.train_indicies]

            self.X_val = X.iloc[self.validation_indicies]
            self.y_val = y.iloc[self.validation_indicies]
            self.costs_val = costs[self.validation_indicies]

        #X_preprocessed, _= preprocessing_function(X)  
        ref_point = np.array([make_ref_point(np.array(self.y_train),np.array(self.train_costs),normalize_churn=True)])
        self.ga_algorithm = define_algorithm(algorithm_name= self.actual_params['GA_algorithm'],
                            pop_size=self.actual_params['population_size'],
                            crossover_op = self.actual_params['crossover_op'],
                            mutation_op= self.actual_params['mutation_op'],n_objeectives=len(self.objectives),
                            ref_points=ref_point
                            ) 
        final_X_train = self.X_train.copy()
        if self.actual_params['winsorize']:
             final_X_train = final_X_train.apply(lambda s: mstats.winsorize(s, limits=[0.025, 0.025])) 
        if self.actual_params['use_rulefit']: 
            print('fitting rulefit')
            clf = RuleFit(rfmode='classify', n_jobs = -1, model_type='lr', tree_generator = copy.deepcopy(self.actual_params["rulefit_base_model"]))
            clf.fit(self.X_train.values.astype(np.float32), self.y_train,feature_names = self.X_train.columns)
            rules = clf.get_rules()
            print('fitting is done!')
            rules = rules[(rules.coef != 0) & (rules.type != 'linear')].sort_values("importance", ascending=False).reset_index()
            self.rules = OrderedDict()
            for rule_index, rule in rules.iterrows():
                if rule_index == self.actual_params['top_rules_count'] : 
                    break
                self.rules[f'rule_{rule_index}_feature'] = rule['rule'] 
                final_X_train[f'rule_{rule_index}_feature'] = self.X_train.apply(lambda row: self.evaluate_rule(row, rule_str = rule['rule']),axis=1)
        if self.actual_params['add_sota_prob']: 
            print('adding prob of merg')
            self.sota_model = copy.deepcopy(self.actual_params['SOTA_model'])
            self.sota_model.fit(self.X_train, self.y_train)
            final_X_train['predicted_prob_of_merge'] = self.sota_model.predict_proba(self.X_train)[:, 1]
        print(final_X_train.head())
    
        self.learned_weights, self.weights_objectives, self.opt_problem = train_MOGA(final_X_train,np.array(self.y_train),np.array(self.train_costs),self.ga_algorithm,self.actual_params['n_gen'], self.objectives)
        #print(self.weights_objectives)
        self.is_fit = True
        if  not(train_indicies is None) : 
            val_predictions = self.predict(self.X_val)
            accs_val, popts_val = evaluate_predictions(val_predictions,self.y_val,self.costs_val,k=0.2) 
            self.best_model_idx = np.argmax(np.array(popts_val))
        else :
            train_predictions = self.predict(self.X_train)
            accs_train, popts_train = evaluate_predictions(train_predictions,self.y_train,self.train_costs,k=0.2) 
            self.best_model_idx = np.argmax(np.array(popts_train))
        
    def predict(self,X) : 
        #X_preprocessed,_ = self.preprocessing_function(X)
        probabilities = self.predict_proba(X)
        predictions = predict(probabilities,self.opt_problem.prediction_threshold)
        #costs = self.prediction_cost_function(X,predictions)
        return predictions



    def predict_proba(self,X) : 
        if not(self.is_fit) : 
            raise NotFittedError('GA is not fitted')
        final_X = X.copy()
        if self.actual_params['winsorize']:
             final_X = X.apply(lambda s: mstats.winsorize(s, limits=[0.025, 0.025])) 
        if self.actual_params['use_rulefit']: 
            for rule_name, rule_str in self.rules.items() : 
                final_X[rule_name] =  X.apply(lambda row: self.evaluate_rule(row, rule_str = rule_str),axis=1)
        if self.actual_params['add_sota_prob']: 
            self.sota_model = copy.deepcopy(self.actual_params['SOTA_model'])
            self.sota_model.fit(self.X_train, self.y_train)
            final_X['predicted_prob_of_merge'] = self.sota_model.predict_proba(X)[:, 1]
        return compute_predictions_probabilities(final_X,self.learned_weights)

    def set_params(self,params) : 
        for param_name,value in params.items():
             self.actual_params[param_name] = value
             self.is_fit = False 
    
    def evaluate_rule(self, row, rule_str): 
        literals = rule_str.split(' & ')
        for literal in literals: 
            splitted_literal = literal.split(' ')
            feature_name, operator, value = splitted_literal[0], splitted_literal[1], splitted_literal[2]
            value = float(value)
            if operator == '<=': 
                res = row[feature_name] <= value
            if operator == '>' : 
                res = row[feature_name] > value
            if not(res) : 
                return 0
        return 1

    
class EARL_model_warpper: 
    def __init__(
                    self,
                    sklearn_model,
                    params = {},
                    objectives = None # never used just for coherency 
                ): 

        
        self.is_fit=False 
        self.sklearn_model = sklearn_model
        self.default_params = {
            'use_rus' : False
        }
        self.params = self.default_params
        for param_name, param_val in params.items(): 
            self.params[param_name] = param_val

    def fit(self,X,y,costs,
            train_indicies = None, 
            validation_indicies = None) : 
        X_final = copy.deepcopy(X)
        y_final = copy.deepcopy(y)
        costs_final = copy.deepcopy(costs)
        if self.params['use_rus']: 
            print('applying RUS')
            rus = RandomUnderSampler()
            X_final, y_final = rus.fit_resample(X_final, y_final)
            costs_final = costs[rus.sample_indices_]
        effort_aware_outcome = y_final*1.0/costs_final
        for z, ev in enumerate(costs) : 
            if ev ==0:
                print('owww zerooooo')
        self.sklearn_model.fit(X_final,effort_aware_outcome)
        self.is_fit = True 
    
    def predict(self,X) : 
        if not(self.is_fit) : 
            raise NotFittedError('SOTA is not fitted')

        predictions = np.array([self.sklearn_model.predict(X)])
        return predictions.T

    def predict_proba(self,X) : 
        if not(self.is_fit) : 
            raise NotFittedError('SOTA is not fitted')

        probabilities = np.array([self.sklearn_model.predict(X)])
        return probabilities.T
class sklearn_model_warpper: 
    def __init__(
                    self,
                    sklearn_model,
                    params = {},
                    objectives = None # never used just for coherency 
                ): 

        
        self.is_fit=False 
        self.sklearn_model = sklearn_model
        self.params = params

    
    def fit(self,X,y,costs,
            train_indicies = None, 
            validation_indicies = None) : 
        self.sklearn_model.fit(X,y)
        self.is_fit = True 
    
    def predict(self,X) : 
        if not(self.is_fit) : 
            raise NotFittedError('SOTA is not fitted')

        predictions = np.array([self.sklearn_model.predict(X)])
        return predictions.T

    def predict_proba(self,X) : 
        if not(self.is_fit) : 
            raise NotFittedError('SOTA is not fitted')

        probabilities = np.array([self.sklearn_model.predict_proba(X)[:,1]])
        print(probabilities)
        return probabilities.T


        
def train_MOGA(data,outcome,costs,algorithm,n_gen = 200,
                objectives = ['recall','churn_cost_predictions']): 
    
    problem = LearnLrWeights(data, outcome, costs,objectives = objectives)
    algorithm = copy.deepcopy(algorithm)
    

    res = minimize(problem,
                    algorithm,
                   ('n_gen', n_gen),
                   seed=1,
                   verbose=True)
    X, F = res.opt.get("X", "F")
    return X,F,res.problem

#main 
def cross_validation(results_path,exp_id,dfs, models
                    ,features=FEATURES,outcome=OUTCOME,
                    nb_folds=10,nb_repetitions=31,
                    ks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) : 
    '''
    This is the main function for cross-validation. It takes as input the data and the models to test.
    It then performs online cross-validation for all the models with the specified repetition number.
    The function saves the results in the results path. 
    '''
    results_path = os.path.join(results_path,exp_id) 
    all_results=[]
    os.makedirs(results_path,exist_ok=True)
    for project_name, (df,preprocessing_metadata) in dfs.items() : 
        org_results = []
        print('Org:',project_name)
        org_path = os.path.join(results_path,project_name)
        os.makedirs(org_path,exist_ok=True)
        pickle.dump(preprocessing_metadata, open(os.path.join(org_path,f'preprocessing_metadata.pkl'), 'wb'))
        df.to_csv(os.path.join(org_path,'all_data.csv'),index=False)
        for model_name, model_data in models.items() : 
            model_results = []
            print('model:',model_name)
            min_repetions = 0
            max_repetions = nb_repetitions
            model_path = os.path.join(org_path,model_name)
            if os.path.isfile(os.path.join(model_path,f"{model_name}.csv")) : 
                cached_model_results = pd.read_csv(os.path.join(model_path,f"{model_name}.csv"))
                max_repetitions_runned = max(cached_model_results['repetition']) + 1 
                if max_repetitions_runned == max_repetions: 
                    print('model already trained')
                    org_results += cached_model_results.to_dict('records')
                    continue
                else : 
                     min_repetions = max_repetitions_runned 
                print(f'running {model_name}')

            for repetition in range(min_repetions, nb_repetitions) :
                print('** Repetation:',repetition)
                for fold in range(1,nb_folds+1):
                    new_row = {
                        'org' : project_name,
                        'model': model_name,
                        'repetition' : repetition,
                        'fold' : fold
                    }
                    print('***fold:',fold)
                    print('fold:',fold)
                    fold_path = os.path.join(model_path,f'repetition{repetition}',f'fold{fold}')
                    os.makedirs(fold_path,exist_ok=True)
                    #checking whether the model is already trained
                    #if it is then no need to re-train
                    if os.path.isfile(os.path.join(fold_path,f'{model_name}.pkl')) : 
                        print('Already done')
                        continue
                    #specifying the training and testing data folds 
                    train_size = df.shape[0] * fold // (nb_folds + 1)
                    test_size = min(df.shape[0] * (fold + 1) // (nb_folds + 1), df.shape[0])
                    df_train = df.iloc[:train_size - 1]
                    df_test = df.iloc[train_size:test_size - 1]
                    #saving train and test data in the experiment metadata
                    df_train.to_csv(os.path.join(fold_path,'train.csv'),index=False)
                    df_test.to_csv(os.path.join(fold_path,'test.csv'),index=False)
                    #preparing Xs and ys for training and testing 
                    x_train, y_train = df.loc[:train_size - 1, features], df.loc[:train_size - 1, outcome]
                    x_test, y_test = df.loc[train_size:test_size - 1, features], df.loc[train_size:test_size - 1, outcome]

                    #computing training and testing costs. Will be used later for MOO
                    train_cost = get_LOC_cost(df.loc[:train_size - 1, features])
                    test_cost = get_LOC_cost(df.loc[train_size:test_size - 1, features])
                    train_indicies = None 
                    val_indicies = None 
                    if not (model_data['validation_selection'] is None) : 
                        train_indicies, val_indicies = model_data['validation_selection'](x_train,y_train,train_cost)

                    #normalizing the data  
                    x_train_preprocessed,_ = preprocess(x_train,preprocessing_metadata=preprocessing_metadata)
                    if ("EALGBM" in model_name) or ("EALR" in model_name):
                        x_train_preprocessed = x_train_preprocessed.drop(columns = ['lines_added', 'lines_deleted' ])

                    #fitting the model
                    start = time.time()
                    model_data['learner'].fit(x_train_preprocessed, y_train, train_cost, train_indicies=train_indicies, validation_indicies=val_indicies)
                    fit_time = time.time() - start

                    #saving the fitted model 
                    pickle.dump(model_data['learner'], open(os.path.join(fold_path,f'{model_name}.pkl'), 'wb'))

                    #preparing the X_test 
                    x_test_preprocessed,_ = preprocess(x_test,preprocessing_metadata=preprocessing_metadata)
                    if ("EALGBM" in model_name) or ("EALR" in model_name):
                        x_test_preprocessed = x_test_preprocessed.drop(columns = ['lines_added', 'lines_deleted' ])

                    #computing predictions and probabilities (i.e., merge likelihood) using the fitted model on the test data   
                    predictions = model_data['learner'].predict(x_test_preprocessed)
                    probabilities = model_data['learner'].predict_proba(x_test_preprocessed)
                       
                    #AUC and recall are not defined for EALGBM and EALR so we compute it for the other models   
                    if model_name != 'EALGBM' and model_name != 'EALR' :
                        test_recalls = [recall(y_test,predictions[:,i]) for i in range(predictions.shape[1]) ]
                        test_aucs = [roc_auc_score(y_test,probabilities[:,i]) for i in range(predictions.shape[1]) ]
                     
                        new_row['best_recall'] = float(np.max(test_recalls))  
                        new_row['median_recall'] = float(np.median(test_recalls))
                        new_row['best_auc'] = float(np.max(test_aucs))  
                        new_row['median_auc'] = float(np.median(test_aucs))  
                    density = y_test/ test_cost
                    for k in ks:
                        accs = []
                        popts = []
                        for i  in range(predictions.shape[1]) : 
                            predict_y = np.array(probabilities[:,i])
                            parameter = np.hstack((np.array([density]).T, np.array([test_cost]).T, np.array([y_test]).T, np.array([predict_y]).T))
                            acc, popt = acc_popt(parameter,k=k)
                            accs.append(acc)
                            popts.append(popt)
                        new_row[f'median_recall_at_{k}'] = float(np.median(accs))
                        new_row[f'max_recall_at_{k}'] = float(np.max(accs))
                        new_row[f'median_popt'] = float(np.median(popts))
                        new_row[f'max_popt'] = float(np.max(popts))
                    new_row['fit_time_seconds'] = float(fit_time)
                    print(os.path.join(fold_path,f'{model_name}_stats.json'))
                    with open(os.path.join(fold_path,f'{model_name}_stats.json'),"w+", encoding="utf8") as f: 
                        print(f.name)
                        json.dump(
                            new_row,
                            f,
                            indent=4,
                            separators=(',', ': '))
                    model_results.append(new_row)
            model_performance_df = pd.DataFrame(model_results)
            model_performance_df.to_csv(os.path.join(model_path,f'{model_name}.csv'),index=False)
            org_results += model_results
        all_results += org_results
        org_performance_df=pd.DataFrame(org_results)
        org_performance_df.to_csv(os.path.join(org_path,f'{project_name}.csv'),index=False)
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(results_path,'all.csv'),index=False)

def main(): 
    #main 
    #setting algos 
    TWO_OBJECTIVE_ALGO = 'NSGA2'
    all_algos = {}

    
    all_algos['EALR'] = {
        'learner' : EARL_model_warpper(sklearn_model=LinearRegression()),
        'validation_selection' : None
    }
    all_algos['EALGBM'] = {
        'learner' : EARL_model_warpper(sklearn_model=get_sota_model_EA()),
        'validation_selection' : None
    }


    all_algos['SOTA_islam'] = {
        'learner' : sklearn_model_warpper(sklearn_model=get_sota_model()),
        'validation_selection' : None
    }
    '''
    for algo in MOGA_ALGOS: 
        model_name=f'{algo}_LR_recall_TNR_churn'
        all_algos[model_name]= {}
        all_algos[model_name]['learner'] = MOGA_LR_warapper(
                        params = {},
                        objectives = ['recall','AUC','normalized_churn_cost_predictions'])
        all_algos[model_name]['learner'].set_params({
            'mutation_op' : PolynomialMutation(eta=20,prob = 1/(len(FEATURES)) ),
            'GA_algorithm':algo,
            'population_size':800,
            'n_gen' : 2400,
            'crossover_op' :  SBX(prob=0.7, eta=15)
        })
        all_algos[model_name]['validation_selection'] = None #lambda X,y,costs: make_validation_set(X,y,costs,validation_size = None,stratify_on_costs=False)
    '''   
    for metrics_pair in [ ['AUC','average_churn_recall']] :  
        model_name=f'{TWO_OBJECTIVE_ALGO}_LR_{metrics_pair[0]}_{metrics_pair[1]}'
        all_algos[model_name]= {}
        all_algos[model_name]['learner'] = MOGA_LR_warapper(
                        params = {},
                        objectives = metrics_pair)
        all_algos[model_name]['learner'].set_params({
            'mutation_op' : PolynomialMutation(eta=20,prob = 1/(len(FEATURES))),
            'GA_algorithm':TWO_OBJECTIVE_ALGO,
            'population_size':800,
            'n_gen' : 2400,
            'use_rulefit' : True,
            'crossover_op' :  SBX(prob=0.7, eta=15)
        })
        all_algos[model_name]['validation_selection'] = None #lambda X,y,costs: make_validation_set(X,y,costs,validation_size = None,stratify_on_costs=False)
    '''
    GA_AUC_recall_and_churn = MOGA_LR_warapper(
                        params = {},
                        objectives = ['average_AUC_and_recall_churn'])

    GA_AUC_recall_and_churn.set_params({
        'mutation_op' : PolynomialMutation(eta=20,prob = 1/(len(FEATURES)) ),
        'GA_algorithm':'GA',
        'population_size':800,
        'n_gen' : 2400,
        'crossover_op' :  SBX(prob=0.7, eta=15)
    })

 
    
    #all_algos['GA_AUC_recall_and_churn']= {}
    #all_algos['GA_AUC_recall_and_churn']['learner'] = GA_AUC_recall_and_churn
    #all_algos['GA_AUC_recall_and_churn']['validation_selection'] = None
    #preparing data 
    '''
    dfs = {}

    for org in ['Eclipse','Libreoffice','Gerrithub']: 
        set_global_vars(org)
        data = GET_DATA()
        print(len(data))
        ready_data, preprocessing_metadata =  prepare_data(data) 
        dfs[org] = (copy.deepcopy(ready_data),copy.deepcopy(preprocessing_metadata))
        
    cross_validation(results_path = os.path.join('.','results'),exp_id='pop,800_GEN,2400_AUC_TPR_churn_no_rulefit',dfs = dfs, models = all_algos
                        ,features=FEATURES,outcome=OUTCOME,
                        nb_folds=10,nb_repetitions=1,ks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99,0.999]) 

if __name__=="__main__":
    #freeze_support()
    main()


