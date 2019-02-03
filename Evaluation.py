# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:28:44 2017

@author: s8350866
"""

import pandas as pd
import numpy as np
import math
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from BI.Manipulation import Feature_Selection as FS

class Model_Evaluation(object):
    '''
    This class is used to check how good is your model. As there is no yes or no answer to this question, 
    in this class there a few different evaluations techniques.
    
    General:
    
    outputpath:                                                                default=''    
    dataframe:                                                                 default=pd.DataFrame()
    target: example - isNesher                                                 default=''
    key: example - PN                                                          default=''
    
    Algorithms:
    
    features:                                                                  default=[] 
    train_df:                                                                  default=pd.DataFrame() 
    pred_df:                                                                   default=pd.DataFrame()
    algorithms:                                                                default=['ld','Ga','Rf','Rf_Grid']
                 
    Precision_Recall:
    
    precall_opt: 'PRC'/'Quantitative '/'Percentage'                            default='PRC'    
    minimum_recall:                                                            default=0.3
    change_in_pred:                                                            default=0.01
    max_attempts:                                                              default=500
    step: How we cut the dataframe. Could mean a precent or amount             default=1000
    cut_loop:                                                                  default=True

    Threshold_Selection:
    
    precall_df:                                                                default=pd.DataFrame()      
    min_threshold_size:                                                        default=30  
    threshold_method: 'MATKA'/'AUC'                                            default='MATKA'
    
    Algorithm_Selection:
    
    best_threshold_precall_dict:                                               default={}  
    pred_column_list: A list even for one column                               default=[]    
    evaluation_methods: A list even for one column                             default=['Recall','Precision','Ratio_between_groups',
                                                                                        'Average_difference','RMSE','MAE','R^2','F_measure','AUC']
    DM:
    run_feature_select:                                                        default=True
    
    STD:
    
    feats_for_std:                                                             default=[]
    std_range: if you want a group smaller than the first and bigger
               than the last add float('inf') and float('-inf')                default=[float('-inf'),0,1,float('inf')]
    std_range_labels:                                                          default=["smaller than 0",0-1,"bigger than 1"]
    '''
    
    def __init__(self,outputpath='',
                 dataframe=pd.DataFrame(),
                 target='',
                 key='', 
                 
                 features=[], 
                 train_df=pd.DataFrame(), 
                 pred_df=pd.DataFrame(),
                 algorithms=['ld','Ga','Rf','Rf_Grid'],

                 precall_opt='PRC',
                 minimum_recall=0.3,
                 change_in_pred=0.01,
                 max_attempts=500,
                 step=1000,
                 cut_loop=True,
                 
                 precall_df=pd.DataFrame(),
                 threshold_method='MATKA',
                 min_threshold_size=30,
                 
                 best_threshold_precall_dict={},
                 pred_column_list=[],
                 evaluation_methods=['Recall','Precision','Ratio_between_groups',
                                   'Average_difference','RMSE','MAE','R^2','F_measure','AUC'],
                 
                 run_feature_select=True,
                 
                 feats_for_std=[],
                 std_range=[float('-inf'),0,1,float('inf')], 
                 std_range_labels=['smaller than 0','0-1','bigger than 1']):
              
        #General
        self.outputpath = outputpath
        self.dataframe = dataframe
        self.target = target       
        self.key = key
        
        #Algorithms
        self.features = features
        self.train_df = train_df
        self.pred_df = pred_df
        self.algorithms = algorithms
        
        #Precision_Recall 
        self.precall_opt = precall_opt
        self.minimum_recall = minimum_recall
        self.change_in_pred = change_in_pred
        self.max_attempts = max_attempts
        self.step = step
        self.cut_loop = cut_loop
        
        #Threshold_Selection (also uses Precision_Recall)
        self.precall_df = precall_df
        self.threshold_method = threshold_method
        self.min_threshold_size = min_threshold_size
        
        #Algorithm_Selection (also uses Threshold_Selection)
        self.best_threshold_precall_dict = best_threshold_precall_dict        
        self.pred_column_list = pred_column_list
        self.evaluation_methods = evaluation_methods
        
        #DM
        self.run_feature_select = run_feature_select
        
        #STD
        self.feats_for_std = feats_for_std
        self.std_range = std_range
        self.std_range_labels = std_range_labels
    
    def Algorithms(self):
        '''This functions runs all the normal algorithms we use in our predictions:
        LDA, GaussianNB and RandomForestClassifier.  For RFC we also run an Hyper parameters run as
        in many cases it improves our results.
        *Notice that running the Hyper parameters RFC might take some time.
        
        Input:
        
            --------------------
            
            train_df(=pd.DataFrame()) 
            pred_df(=pd.DataFrame()) 
            features(=[]) 
            target(='')
            algorithms(=['ld','Ga','Rf','Rf_Grid']) --> You can run only some of them if you want
            outputpath(='') --> If left as '' than no file will be created
            
            --------------------
            
        Output:
        
            An updated dataframe which contains our prediction columns
            
        '''
        
        self.pred_column_list = self.algorithms
        
        Rf_parameters = {'min_samples_leaf':np.arange(0.05,0.1,0.05),'min_samples_split':range(2,100,2)}
        
        dict_algs = {'ld' : LDA(priors=[0.5,0.5]),
                     'Ga' : GaussianNB(),
                     'Rf' : RandomForestClassifier(random_state=0),
                     'Rf_Grid' : GridSearchCV(RandomForestClassifier(random_state=0),Rf_parameters)}
                     
        for alg in self.algorithms:
            alg_fited = dict_algs[alg].fit(self.train_df[self.features],self.train_df[self.target]) 
            
            self.pred_df[alg]= alg_fited.predict_proba(self.pred_df[self.features])[:,1]
        
        if self.outputpath!='':
            self.pred_df.to_csv(self.outputpath+'_preds.csv',index=False,encoding='cp1255')
        return self.pred_df 
        
    def Precision_Recall(self):
        '''
        There are three methods to check precision and recall: PRC, Quantitative  and Percentage

        PRC --> Checks every prediction in an ascending order and calculate it's precision and recall

        Quantitative --> Cuts the dataframe so will be left with a certain amount of people and than checks the 
        precision and recall of this group

        Percentage --> Same as Quantitative  only that we cut by Percentage
        
        Input:
        
            --------------------
            
            dataframe(=empty dataframe)
            target(='')
            key(='')
            pred_column_list(=[]) 
            precall_opt(='PRC') --> Can be: 'PRC'/'Quantitative'/'Percentage'
            outputpath(='') --> If left as '' than no file will be created
            
            --------------------
            
            <>If precall_opt is 'Quantitative'/'Percentage' than you can change the following arguments aswell: 
              
              step(=1000) 
              cut_loop(=True)
              
            --------------------
        
        Output: 
        
            A dataframe - {"Algorithm","Threshold","Precision","Recall"}
        '''

        all_thresholds = pd.DataFrame() #Keeps all the scores for the algorithm
        if self.precall_opt=='PRC':
            for alg in self.pred_column_list:
                all_thresholds_spec = pd.DataFrame()
                precisions, recalls, thresholds = metrics.precision_recall_curve(self.dataframe[self.target], self.dataframe[alg])
                thresholds = np.append(thresholds,1) 
                all_thresholds_spec["Threshold"] = thresholds
                all_thresholds_spec["Precision"] = precisions
                all_thresholds_spec["Recall"] = recalls
                all_thresholds_spec["Algorithm"] = alg

                sizes_df = pd.pivot_table(self.dataframe,index=alg,values=self.key,aggfunc=np.size).reset_index()
                sizes_df.sort_values(alg,ascending=False,inplace=True)                
                sizes_df['real_size']=sizes_df[self.key].cumsum()
                if 1 not in sizes_df[alg]:
                    sizes_df = sizes_df.append({alg:1,self.key:0,'real_size':0},ignore_index=True)
                
                all_thresholds_spec = all_thresholds_spec.merge(sizes_df,left_on="Threshold",right_on=alg,how='left')
                del all_thresholds_spec[self.key]
                del all_thresholds_spec[alg]

                all_thresholds = all_thresholds.append(all_thresholds_spec,ignore_index=True)

        elif self.precall_opt=='Quantitative ' or self.precall_opt=='Percentage':
            target1_full = self.dataframe[self.dataframe[self.target]==1].shape[0] #All the target=1 people
            if self.precall_opt=='Quantitative ':
                maxi = self.dataframe.shape[0]
            if self.precall_opt=='Percentage':
                maxi = 100
            for cut in range(self.step,maxi,self.step):
                for alg in self.pred_column_list:
                    df_cut = self.dataframe.sort_values([alg],ascending=False)
                    
                    if self.precall_opt=='Quantitative ': #Cutting by the amount of people
                        df_cut2 = df_cut.head(cut)
                        cut2 = cut
                    elif self.precall_opt=='Percentage': #Calculating the precent and then what amount is it
                        precent = self.dataframe.shape[0]/100
                        cut2 = int(cut*precent)
                        df_cut2=df_cut.head(cut2)

                    target1 = df_cut2[self.dataframe[self.target]==1].shape[0] #All the target=1 people in this cut
                    target0 = df_cut2.shape[0] - target1
                    totalRec = target1 + target0 #All the people
                    
                    table = pd.pivot_table(df_cut2,columns=[self.target],values=self.key,index=alg,aggfunc='count',fill_value=0).reset_index() #index  alg  target=0  target=1 
                    table.sort_values([alg],ascending=False,inplace=True) #Sorting so when we go row by row the prediction will increase
        
                    precision = target1 / totalRec
                    recall = target1 / target1_full
                    
                    row_data = {"Algorithm":alg,"Threshold":cut2,"Precision":precision,"Recall":recall,'real_size':cut2}
                    if precision+recall>0:
                        all_thresholds = all_thresholds.append(row_data,ignore_index=True)
                if self.cut_loop==0:
                    break
        
        if self.outputpath!='':
            all_thresholds.to_csv(self.outputpath+'_precalls.csv',index=False,encoding='cp1255')
        return all_thresholds
            
            
    def Threshold_Selection(self):
        '''
        There are two methods to select the best threshold: MATKA and AUC
        
        MATKA --> Checks every prediction in an ascending order and if the precision is better than the
        last row and the recall is not too small he will choose it as the best threshold.
        
        AUC --> Calculate the graph of our precision and recall, and checks each point on the graph(which is a different threshold),
        to see which is the closest to (0,1)
        
        Input: 
        
          <>If threshold_method is 'MATKA' than you need to provide the following arguments:
            
          --------------------
            
            dataframe(=empty dataframe) 
            target(='')
            key(='')
            pred_column_list(=[])  
            minimum_recall(=0.3)
            change_in_pred(=0.01) 
            max_attempts(=500)
            min_threshold_size(=30)
            threshold_method(='MATKA') --> Can be: 'MATKA'/'AUC'
            outputpath(='') --> If left as '' than no file will be created
            precall_df(=empty dataframe)
        
          --------------------
          
          <>If precall_df is an empty dataframe than you can change tose following arguments aswell:
          
            step(=1000) 
            cut_loop(=True)
            precall_opt(='PRC') --> Can be: 'PRC'/'Quantitative'/'Percentage'
            
          --------------------
                
          <>If threshold_method is 'AUC' than you only(!) need to provide the following arguments:
        
            dataframe(=empty dataframe) 
            target(='') 
            pred_column_list(=[]) 
            threshold_method(='MATKA') --> Can be: 'MATKA'/'AUC'
            outputpath(='') --> If left as '' than no file will be created
            
          --------------------
                
        Output: 
        
            A dictionary - {Algorithm: {"Threshold":,"Precision":,"Recall":,"Size":}}
        '''
        
        best_thresholds ={}
        
        if self.threshold_method=='MATKA':
            if self.precall_df.empty == True:
                self.precall_df = Model_Evaluation.Precision_Recall(self)

            for alg in self.pred_column_list:
                
                best_threshold = 0  
                Precision = 0
                Recall = 0
                n = 0
                i = 1
                iPrec = 0
                
                df_alg = self.precall_df[self.precall_df['Algorithm']==alg]
                df_alg.sort_values(['Precision'],ascending=True,inplace=True) #Sorting so when we go row by row the prediction will increase
                df_alg = df_alg[df_alg['Recall']>=self.minimum_recall] #If the recall is too low there is now need to check this row        
                
                head_list = df_alg.columns.tolist()
                threshold_ind = head_list.index('Threshold')+1
                precision_ind = head_list.index('Precision')+1
                recall_ind = head_list.index('Recall')+1
                #algorithm_ind = head_list.index('Algorithm')+1
                real_size_ind = head_list.index('real_size')+1

                for row in df_alg.itertuples(): 
                    alg_prec = row[precision_ind]  
                    recall = row[recall_ind]
                    size = row[real_size_ind]
                    if  alg_prec >= Precision and recall >= self.minimum_recall and size >= self.min_threshold_size: #Checks if the precision is better than previos and the recall is still ok                      
                        n = 0                        
                        Precision = alg_prec
                        best_threshold = row[threshold_ind]
                        Recall = recall
                        Size = size
                        
                        #Because when we increase our precision we decrease our recall, we need to check that its worth it.
                        #So if we had a really small increase in precision for a long time we stop 
                        if i == 1: 
                            iPrec = Precision
                        if i == self.max_attempts:
                            if Precision - iPrec < self.change_in_pred:
                                break
                            else:
                                iPrec = Precision
                            i = 1
                        i += 1 
                    #We also want to stop if we haven't got a better result for a long time or if we already got too small recall
                    else: #If the precision is not better than last row or if the recall is too small
                        if  n < self.max_attempts and alg_prec < Precision and recall >= self.minimum_recall:                        
                            n += 1
                            i = 1
                        else:
                            if n >= self.max_attempts or recall < self.minimum_recall:
                                break
                best_thresholds[alg] = {}
                best_thresholds[alg]["Threshold"] = best_threshold  
                best_thresholds[alg]["Precision"] = Precision
                best_thresholds[alg]["Recall"] = Recall
                best_thresholds[alg]["Size"] = Size

        if self.threshold_method=='AUC':
            for alg in self.pred_column_list:
                best_thresholds[alg] = {}
                fp_rate, tp_rate, thresholds = metrics.roc_curve(self.dataframe[self.target], self.dataframe[alg]) #A built function that returns three lists: thresholds, false positive and true posotive               
                #False positive are all the people we sayed that are going to be target=1 but are actually target=0 divided by all the real 0
                #True positive = Recall
                #Basiclly we get graph that X is FP and Y is TP and the names of the points are the thresholds                    
                th=0
                Mindis=-1
                for i in range(0,len(thresholds)):
                    if thresholds[i] > 0:
                        if fp_rate[i]!=0 or tp_rate[i]!=0:
                            dis=math.hypot(fp_rate[i]-0, tp_rate[i]-1) #Calculating the distance between our point(threshold) to (0,1)                       
                            if Mindis==-1 or dis<Mindis: #If its closer than previos threshold we take it
                                th=thresholds[i]
                                Mindis=dis

                best_thresholds[alg]["Threshold"] = th 
                df_check = self.dataframe[self.dataframe[alg]>=th]
                Precision = df_check[df_check[self.target]==1].shape[0]/df_check.shape[0]
                Recall = df_check[df_check[self.target]==1].shape[0]/self.dataframe[self.dataframe[self.target]==1].shape[0]
                best_thresholds[alg]["Precision"] = Precision
                best_thresholds[alg]["Recall"] = Recall
                best_thresholds[alg]["Size"] = df_check.shape[0]

        #Transforming the dictionary to a dataframe so we can export it to csv
        best_thresholds_df=pd.DataFrame()
        for key in best_thresholds:
            row = {'Algorithm':key,'Precision':best_thresholds[key]['Precision'],
                   'Recall':best_thresholds[key]['Recall'],
                   'Threshold':best_thresholds[key]['Threshold'],
                   'Size':best_thresholds[key]['Size']}
            best_thresholds_df = best_thresholds_df.append(row,ignore_index=True)
            
        if self.outputpath!='':
            best_thresholds_df.to_csv(self.outputpath+'_thresholds.csv',index=False,encoding='cp1255')
        return best_thresholds

    
    def Algorithm_Selection(self):
        '''
        There are a few methods to decide which algorithm is the best: 
        Precision, Recall, F_measure and Ratio_between_groups - Need a threshold value that we get from Threshold_Selection
        RMSE, R^2, Average_difference and AUC - Don't need a value
        
        Input: 
                
          --------------------
          
            dataframe(=empty dataframe) 
            target(='')
            key(='')
            pred_column_list(=[])
            evaluation_methods(=['Recall','Precision','Ratio_between_groups','Average_difference','RMSE','MAE','R^2','F_measure','AUC']) --> You can ask for parts of this list
            outputpath(='') --> If left as '' than no file/s will be created
            best_threshold_precall_dict(=empty dictionary)
            
          --------------------
          
          <>If best_threshold_precall_dict is an empty dictionary than you can change the following arguments aswell:
            
            minimum_recall(=0.3)
            change_in_pred(=0.01) 
            max_attempts(=500)
            min_threshold_size(=30)
            threshold_method(='MATKA') --> Can be: 'MATKA'/'AUC'
            precall_df(=empty dataframe)
        
          --------------------
          
          <>If precall_df is an empty dataframe than you can change the following arguments aswell:
            
            step(=1000) 
            cut_loop(=True)
            precall_opt(='PRC') --> Can be: 'PRC'/'Quantitative'/'Percentage'
            
          --------------------

        Output: 
        
            A dctionary - {'Precision':,'Recall':,'RMSE':,'MAE':,'R^2':,'F_measure':,'AUC':,
                           'Ratio_between_groups':,'Average_difference':}
        '''

        weight_dict = {}
        for alg in self.pred_column_list:
            weight_dict[alg] = {}
            
            if self.best_threshold_precall_dict == {}:
                self.best_threshold_precall_dict = Model_Evaluation.Threshold_Selection(self) #Getting the best thresholds 
            if 'Recall' in self.evaluation_methods:  
                recall =  self.best_threshold_precall_dict[alg]['Recall'] #Getting the best recall we got
                weight_dict[alg]['Recall'] = recall
            if 'Precision' in self.evaluation_methods: 
                precision =  self.best_threshold_precall_dict[alg]['Precision'] #Getting the best precision we got  
                weight_dict[alg]['Precision'] = precision
            if 'F_measure' in self.evaluation_methods:
                f_measure = (2*precision*recall)/(precision+recall)
                weight_dict[alg]['F_measure'] = f_measure
            if 'Ratio_between_groups' in self.evaluation_methods: 
                try:
                    threshold =  self.best_threshold_precall_dict[alg]['Threshold'] #Getting the best threshold we got
                    alg1 = self.dataframe[self.dataframe[alg]>=threshold].shape[0] #Who we consider as 1
                    alg1_target1 = self.dataframe[(self.dataframe[alg]>=threshold)&(self.dataframe[self.target]==1)].shape[0] #Who we predict as 1 and his target is 1
                    alg1_target_rate=alg1_target1/alg1 #The target rate in the group we predict as 1
                    alg0 = self.dataframe[self.dataframe[alg]<threshold].shape[0] #Who we consider as 0
                    alg0_target1 = self.dataframe[(self.dataframe[alg]<threshold)&(self.dataframe[self.target]==1)].shape[0] #Who we predict as 0 and his target is 1
                    alg0_target_rate=alg0_target1/alg0 #The target rate in the group we predict as 0
                    ration_between_groups=alg1_target_rate/alg0_target_rate
                    weight_dict[alg]['Ratio_between_groups'] = ration_between_groups
                except ZeroDivisionError:
                    weight_dict[alg]['Ratio_between_groups'] = 'ZeroDivisionError'
            if 'Average_difference' in self.evaluation_methods:
                target1 = self.dataframe[self.dataframe[self.target]==1] #Everyone the as target 1
                target0 = self.dataframe[self.dataframe[self.target]==0] #Everyone the as target 0
                target1_mean = target1[alg].mean() #The algorithm prediction average for people with target 1
                target0_mean = target0[alg].mean() #The algorithm prediction average for people with target 0
                try:
                    average_difference= target1_mean/target0_mean
                except ZeroDivisionError:
                    average_difference = 'ZeroDivisionError'
                weight_dict[alg]['Average_difference'] = average_difference
            if 'RMSE' in self.evaluation_methods:
                rmse = np.sqrt(mean_squared_error(self.dataframe[self.target],self.dataframe[alg]))
                weight_dict[alg]['RMSE'] = rmse
            if 'MAE' in self.evaluation_methods:
                mae = np.absolute(self.dataframe[self.target]-self.dataframe[alg]).mean()
                weight_dict[alg]['MAE'] = mae
            if 'R^2' in self.evaluation_methods:
                r2 = metrics.r2_score(self.dataframe[self.target],self.dataframe[alg])
                weight_dict[alg]['R^2'] = r2
            if 'AUC' in self.evaluation_methods:
                fp_rate, tp_rate, thresholds = metrics.roc_curve(self.dataframe[self.target], self.dataframe[alg]) #A built function that returns three lists: thresholds, false positive and true posotive
                #False positive are all the people we sayed that are going to be target=1 but are actually target=0 divided by all the real 0
                #True positive = Recall
                #Basiclly we get graph that X is FP and Y is TP and the names of the points are the thresholds    
                weight_dict[alg]['AUC'] = metrics.auc(fp_rate, tp_rate) #Calculates the area under the ROC curve                 
            
        #Transforming the dictionary to a dataframe so we can export it to csv
        weight_df=pd.DataFrame()
        for alg in list(weight_dict.keys()):
            row = {'Algorithm':alg}
            for val_meth in weight_dict[alg]:
                row[val_meth] = weight_dict[alg][val_meth]
            weight_df = weight_df.append(row,ignore_index=True)
        
        if self.outputpath!='': #Save our weights dataframe
            weight_df.to_csv(self.outputpath+'_algorithms.csv',index=False,encoding='cp1255')
            
        #Choosing the best algorithm by: AUC --> Precision --> F_measure --> Recall
        best_alg = self.Algorithm_Selection_helper(weight_df)
        
        return best_alg, weight_dict
    
    def Algorithm_Selection_helper(self,weight_df): 
        parameter_list = ['AUC','Precision','F_measure','Recall']
        for parameter in parameter_list:
            if parameter != parameter_list[-1]: #Checking if it is our last parameter
                new_col_name = 'Diff_from_best_' + str(parameter)
                weight_df = weight_df.sort_values(parameter,ascending=False).reset_index(drop=True) #Sorting our dataframe from the biggest Parameter to the smallest
                weight_df[new_col_name] = weight_df[parameter].loc[0] - weight_df[parameter] #Calculating the difference of each Parameter value from the best Parameter value
                weight_df = weight_df[weight_df[new_col_name]<=0.03].reset_index(drop=True) #Keeping only the algorithms that their Parameter is not smaller in more than 0.03 from our best Parameter(includes our best algorithm in this Parameter)
                
                if weight_df.shape[0]==1:#If there are no algorithms that their Parameter is close to the best Parameter
                    best_alg = weight_df['Algorithm'].iloc[0] #Choosing the algorithm from our first row
                    break
            else: #If it is our last parameter we just take the best algorithm in this parameter
                weight_df = weight_df.sort_values(parameter,ascending=False).reset_index(drop=True) #Sorting our dataframe from the biggest Parameter to the smallest
                best_alg = weight_df['Algorithm'].iloc[0] #Choosing the algorithm from our first row
                
        return best_alg
    
    def DM(self):
        '''This function runs a full prediction, feature selection and algorithm selection for your database. In general you can run while giving only the
        essential parameters and it will use all the default values. If you want to edit the way the function works you can change any of
        the internal functions parameters as you would like.
        *This does not apply to the feature selection, so if you want to choose them differently just do not run it.
        If you don't change the default parameters the code will run the algorithms: LDA, GaussianNB and RandomForestClassifier.  For RFC we also run an Hyper parameters run.
        The precision recall will run by PRC, the threshold selection will be by MATKA and the algorithm selection will
        run by: AUC --> Precision --> F_measure --> Recall
        
        
        The following input is the essential parameters needed so you can run the function
        Input:
        
            --------------------
            
            key(='')
            train_df(=pd.DataFrame()) 
            pred_df(=pd.DataFrame()) 
            features(=[]) 
            target(='')
            algorithms(=['ld','Ga','Rf','Rf_Grid']) --> You can run only some of them if you want
            run_feature_select(=True)
            outputpath(='') --> If left as '' than no file will be created
            
            --------------------
            
        Output:
        
            The dataframe with a prediction colum with the best algorithm'''
        
        if self.run_feature_select:
            obj = FS(train_df=self.train_df, 
                     target=self.target, 
                     features=self.features, 
                     amount_of_features=len(self.features))
            feat_rank_df = obj.Select()[0]
            feat_rank_df['rank_average'] = (feat_rank_df['Entropy']+feat_rank_df['ForestClassifier']+feat_rank_df['SelectKBest'])/3
            feat_rank_df.sort_values('rank_average', ascending=True, inplace=True)
            if feat_rank_df.shape[0]>=10:
                feat_rank_df = feat_rank_df.head(10)  
            self.features = list(feat_rank_df['Feature'].unique())

        self.dataframe = self.Algorithms()
        best_alg = self.Algorithm_Selection()[0]
        
        threshold =  self.best_threshold_precall_dict[best_alg]['Threshold']
        
        self.dataframe['Prediction_'+best_alg] = np.where(self.dataframe[best_alg]>=threshold,1,0)
        
        return self.dataframe
    
    def STD(self):
        '''
        Input: 
        
            --------------------
            
            dataframe(=empty dataframe) 
            target(='')
            key(='')
            std_range(=[float('-inf'),0,1,float('inf')]) 
            std_range_labels(=['smaller than 0','0-1','bigger than 1']) 
            feats_for_std(=[])
            outputpath(='') --> If left as '' than no file/s will be created
            
            Exemple for std_range - [0,1,2] will give us the groups: smaller then 0, between 0 to 1, between 1 to 2 and bigger than 2
        
            --------------------
                
        Output: 
        
            This function returns a Dataframe with three columns - the column name, the group and the target Percentage
        '''

        std_data = pd.DataFrame()
        for feat in self.feats_for_std: #For each column we are doing the std on
            df_feat = pd.DataFrame()
            df_feat[self.key] = self.dataframe[self.key]
            df_feat['Target'] = self.dataframe[self.target]
            df_feat['Feature'] = feat
            df_feat['std'] = (self.dataframe[feat]-self.dataframe[feat].mean())/self.dataframe[feat].std() #(value-column average)/column std
            df_feat['Group'] = pd.cut(df_feat['std'],bins=self.std_range,labels=self.std_range_labels) #Cutting the std column to groups that we chose
            df_feat['Target rate'] = 0
            for grp in df_feat['Group'].unique(): #For each group in this column
                df_feat_grp = df_feat[df_feat['Group']==grp]
                target_rate = df_feat_grp[df_feat_grp['Target']==1].shape[0]/df_feat_grp.shape[0] #We are calculating the target=1 rate in this group
                df_feat['Target rate'] = np.where(df_feat['Group']==grp,target_rate,df_feat['Target rate']) #We are updating this column after each group
            del df_feat['Target']
            std_data = std_data.append(df_feat,ignore_index=True)
            
        if self.outputpath!='':
            std_data.to_csv(self.outputpath+'_STD.csv',index=False,encoding='cp1255')
        return std_data
            