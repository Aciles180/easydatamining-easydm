# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:30:10 2017

@author: s8350866
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
import BI
#from BI.Evaluation import Model_Evaluation as me

class Feature_Selection(object):
    '''
    In this class there are functions that are used to select the most important features or to change
    the features values.
    
    outputpath:                                                                default=''
    train_df:                                                                  default=pd.DataFrame()
    target:                                                                    default=''
    features:                                                                  default=[]
    amount_of_features:                                                        default=len(features)
    feature_selection_methods:                                                 default=['SelectKBest','Entropy','ForestClassifier']
    '''
    
    def __init__(self,outputpath='',
                 train_df=pd.DataFrame(),
                 target='',
                 features=[], 
                 amount_of_features=0,
                 feature_selection_methods=['SelectKBest','Entropy','ForestClassifier']):
        
        #General             
        self.outputpath=outputpath
        self.train_df=train_df
        self.target=target
        
        #Select
        self.features=features
        self.amount_of_features=len(features)
        self.feature_selection_methods=feature_selection_methods
        
    
    def Select(self):
        '''
        Feature selection methods: SelectKBest, Entropy and ForestClassifier
        
        SelectKBest --> Built in function from sklearn
        ForestClassifier --> Built in function from sklearn
        Entropy --> Calculating the features weights by the Entropy method
        
        Input: 
        
            --------------------
            
            feature_selection_methods(=['SelectKBest','Entropy','ForestClassifier']) --> Can be: 'SelectKBest'/'Entropy'/'ForestClassifier'
            train_df(=empty dataframe)
            target(='') 
            features(=[]) 
            outputpath(='') --> If left as '' than no file will be created
            
            --------------------
            
            <>If feature_selection_methods is SelectKBest than we need the following arguments aswell:
        
              amount_of_features(=len(features))
              
            --------------------

        Output: 
        
            There are two returned outputs:
                1. A dataframe - a column of the feature name and columns for each selection method ranks
                2. A dictionary that the keys are the selection method and the value is a descending list of
                   all the features.
            
            Another output:
            A csv with a column of the feature name and columns for each selection method ranks
        '''
        
        feature_rank_dict={}
        df_feat = pd.DataFrame() 
        df_feat['Feature'] = self.features
        if 'SelectKBest' in self.feature_selection_methods:
            df_skb = pd.DataFrame()
            feature_rank_dict['SelectKBest']=[]
            
            #Choosing the self.amount_of_features best features for our target
            test = SelectKBest(score_func=f_classif, k=self.amount_of_features)
            fit = test.fit(self.train_df[self.features], self.train_df[self.target])
            np.set_printoptions(precision=3)
                    
            #Getting the selected features position in the database
            n = 1
            l = []
            for i in list(fit.get_support(indices=False)):
                if i == True:
                    l.append(n)
                n += 1
            
            #According to their position - finding the features names
            f = 1
            for feat in self.features:
                if f in l:
                    dic={'Feature':feat, 'SelectKBest':fit.scores_[f-1]}
                    df_skb = df_skb.append(dic,ignore_index=True)
                f += 1
                
            df_skb.sort_values('SelectKBest',ascending=False,inplace=True)
            for feat in df_skb['Feature']:
                feature_rank_dict['SelectKBest'].append(feat)
                
            df_skb['SelectKBest'] = range(1,df_skb.shape[0]+1)
            df_feat = df_feat.merge(df_skb,on='Feature',how='left')
        
        if 'ForestClassifier' in self.feature_selection_methods:
            df_fc = pd.DataFrame()
            feature_rank_dict['ForestClassifier']=[]

            forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
            forest.fit(self.train_df[self.features],self.train_df[self.target])
            importances = forest.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            rank=1
            for a in indices:
                feat = self.features[a]
                feature_rank_dict['ForestClassifier'].append(feat)
                dic={'Feature':feat,'ForestClassifier':rank}
                df_fc = df_fc.append(dic,ignore_index=True)
                rank+=1
                
            df_feat = df_feat.merge(df_fc,on='Feature',how='left')
            
        if 'Entropy' in self.feature_selection_methods:
            feature_rank_dict['Entropy']=[]
            df_ent = pd.DataFrame(columns = ['Feature','Entropy'])
            for column in self.features:
                df_col = pd.DataFrame(columns = ['Unival','Entropy'])
                if self.train_df[column].unique() != []:
                    for val in self.train_df[column].unique(): 
                        size = self.train_df[self.train_df[column]==val].shape[0]/self.train_df[column].shape[0]   
                        try:
                            nesher_count = self.train_df[(self.train_df[column]==val)&(self.train_df[self.target]==1)].shape[0]/self.train_df[self.train_df[column]==val].shape[0]  
                            not_nesher_count = self.train_df[(self.train_df[column]==val)&(self.train_df[self.target]==0)].shape[0]/self.train_df[self.train_df[column]==val].shape[0] 
                        except ZeroDivisionError:
                            nesher_count = 0
                            not_nesher_count = 0
                            
                        weight = -(nesher_count*np.log2(nesher_count)+not_nesher_count*np.log2(not_nesher_count))  
                        row = {'Unival':column,'Entropy':weight,'size':size}
                        df_col = df_col.append(row,ignore_index=True)
                        df_col['real_weight'] = df_col['size']*df_col['Entropy']
                else:
                    df_col['real_weight'] = 0
                col_weight = np.sum(df_col['real_weight'])
                row2 = {'Feature':column, 'Entropy':col_weight}
                df_ent = df_ent.append(row2,ignore_index=True)
            df_ent.sort_values(['Entropy'],ascending=True,inplace=True)
            df_ent['Entropy'] = range(1,df_ent.shape[0]+1)
                
            df_feat = df_feat.merge(df_ent,on='Feature',how='left')
            
            for feat in df_ent['Feature']:
                feature_rank_dict['Entropy'].append(feat)
                
        if self.outputpath != '':
            df_feat.to_csv(self.outputpath,index=False,encoding='cp1255')
        return df_feat, feature_rank_dict
                
    
class Replace_columns(object):
    '''
    Replaces columns with different values and returns an updated Dataframe

    General:
    
    outputpath:                                                                default=''
    dataframe:                                                                 default=pd.DataFrame()
    target:                                                                    default=''
    key:                                                                       default=''
    replace_columns:                                                           default=True
    
    Groups:
    
    columns_for_groups:                                                        default=[]
    target_rate_min_diff:                                                      default=0.05 
    min_target_shape:                                                          default=30
    
    Normalization:
    
    col_for_method:                                                            default={}
    std_range:                                                                 default=[float('-inf'),0,1,float('inf')] 
    std_range_labels:                                                          default=['smaller than 0','0-1','bigger than 1'] 
    wanted_average:                                                            default=0
    wanted_std:                                                                default=0

    WTF:
    
    wtf_prefix:                                                                default='WTF'
    add_boolwtf:                                                               default=False
    min_wtf_target_size:                                                       default=1.0
    influencing_wtfs:                                                          default=3    
    '''
    
    def __init__(self,
                 outputpath='',
                 dataframe=pd.DataFrame(),
                 target='', 
                 key='',
                 columns_for_groups=[],
                 target_rate_min_diff=0.05, 
                 min_target_shape=30, 
                 replace_columns=True,
                 col_for_method={},
                 std_range=[float('-inf'),0,1,float('inf')], 
                 std_range_labels=['smaller than 0','0-1','bigger than 1'], 
                 wanted_average=0, 
                 wanted_std=0,
                 wtf_prefix='WTF',
                 add_boolwtf=False,
                 min_wtf_target_size=1.0,
                 influencing_wtfs=3):
               
        #General
        self.dataframe = dataframe
        self.key = key
        self.target = target
        self.replace_columns = replace_columns
        
        #Groups
        self.columns_for_groups = columns_for_groups
        self.target_rate_min_diff = target_rate_min_diff
        self.min_target_shape = min_target_shape
        
        #Normalization        
        self.col_for_method = col_for_method
        self.std_range = std_range
        self.std_range_labels = std_range_labels
        self.wanted_average = wanted_average
        self.wanted_std = wanted_std
        
        #WTF
        self.outputpath = outputpath
        self.WTF_prefix = wtf_prefix
        self.add_boolwtf = add_boolwtf
        self.min_wtf_target_size = min_wtf_target_size
        self.influencing_wtfs = influencing_wtfs
    
        
    def Groups(self):
        '''
        Dividing into groups according to the average of the target and combining small groups
        
        Input:
        
            --------------------
            
            dataframe(=empty dataframe)
            target(='')
            columns_for_groups(=[])
            target_rate_min_diff(=0.05) 
            min_target_shape(=30)
            replace_columns(=True)
            
            --------------------
            
        Output:
        
            1) An updated dataframe - It can contain only the new column or also keep the old column as "column name"+"_original"
            2) A decoding dataframe - [column(name),group(ex-"7-9"),group_temp(ex-"7"),group_final(an int so it will be an ascending series)]
        '''
        
        for col in self.columns_for_groups: #Which columns to group - can only be int columns
            self.dataframe[col+'_original'] = self.dataframe[col]
            self.dataframe[col] = round(self.dataframe[col],1)
            col_dict = {}
            col_values = self.dataframe[col].unique() #Getting all the unique values in that column
            col_values.sort()
            for val in col_values:
                #Getting all the people with that value and target=1 and divide it by all the people with that value
                target_precent = self.dataframe[(self.dataframe[col]==val)&(self.dataframe[self.target]==1)].shape[0]/(self.dataframe[self.dataframe[col]==val]).shape[0]
                col_dict[val] = target_precent #Adding to our dictionary
            groups_dict = {} #{goup name:[value1,value2...]}
            time = 0 #Indicator for which value are we on - the first, the second...
            last_unique_val=''
            col_values.sort() #So it will go from the smallest to the biggest
            for unique_val in col_values:
                val_name = str(unique_val)
                if time == 0: #If its the first(=smallest) value
                    old_precent = col_dict[unique_val] #Remmember its target rate
                    old_group = val_name #Remmember the value as a group name
                    groups_dict[val_name] = [unique_val] #Save it in a dictionary for later
                    
                else: #From the second value and so on
                    #We will combine this value with the previos group if we answer one those three conditions:
                    #1. If the current value target rate close to the first one
                    #2. If the current value number of people is too small
                    #3. If the previos value number of people is too small
                    if np.abs(col_dict[unique_val]-old_precent)<=self.target_rate_min_diff or self.dataframe[self.dataframe[col]==unique_val].shape[0]<self.min_target_shape or self.dataframe[self.dataframe[col]==last_unique_val].shape[0]<self.min_target_shape:
                        old_group_name = old_group.split('_')[0]
                        groups_dict[old_group_name + "_" + val_name] = groups_dict.pop(old_group) #Replacing the old group name in the dictionary
                        groups_dict[old_group_name + "_" + val_name].append(unique_val) #Adding our current value to the dictionary
                        old_group = old_group_name + "_" + val_name #Saving the new last group name
                        old_precent = (col_dict[unique_val]+old_precent)/2 #Calculating the new target rate for this group
                    else: #If we dont answer anny of the conditions we start a new group
                        groups_dict[val_name] = [unique_val]
                        old_group = val_name
                        old_precent = col_dict[unique_val]
                    
                last_unique_val=unique_val          
                time += 1
    
            for key in groups_dict: 
                for key2 in groups_dict[key]:
                    self.dataframe[col]=self.dataframe[col].replace(key2,key) #Replcing the value with the group name it is in
        
            if self.replace_columns == 1:
                del self.dataframe[col+'_original']
         
        #Adding the proups int column
        #We cant use a string column for most of our prediction algorithms so we change it to an ascending series
        group_decode_df = pd.DataFrame()#The decoding of the new ascending values
        for col in self.columns_for_groups: #For every column we changed to string groups
            vals = self.dataframe[col].unique() #Getting all the unique grouped values
            milon = {}
            for val in vals: 
                true_val = val.split('_')[0] #Getting the lower number from the string so '7-9' will return '7' and '10' will return '10'
                milon[val] = str(float(true_val)) #Adding the string val as a key and the lower number as the value
            self.dataframe[col] = self.dataframe[col].map(milon) #Replacing the column from the strings to only the lower number
            
            n = 1
            milon2 = {}
            x = list(map(float,list(self.dataframe[col].unique()))) #A list of all the grouped strings lower number after we made sure they are floats
            x.sort() 
            for val2 in x:
                milon2[str(val2)] = n
                n+=1
            self.dataframe[col] = self.dataframe[col].map(milon2) #Replacing the column again so [1,5,8,200] will change to [1,2,3,4]
            
            #Summary of wht we did - [1,2,3,4,....,10] -> [1-4,5,6,8-10] -> [1,5,6,8] -> [1,2,3,4] 
            
            #Creating the specific column group decoding dataframe
            specific_group_decode_df = pd.DataFrame()
            specific_group_decode_df['group'] = list(milon.keys()) #All the string group values - '7-9'
            specific_group_decode_df['group_temp'] = specific_group_decode_df['group'].map(milon) #Only the lowe number - '7'
            specific_group_decode_df['group_final'] = specific_group_decode_df['group_temp'].map(milon2) #The int from the ascending series
            specific_group_decode_df['column'] = col #The column's name           
            group_decode_df = group_decode_df.append(specific_group_decode_df, ignore_index=True) #Adding the specific dataframe to our full column decoding dataframe
            
        return self.dataframe, group_decode_df
    
    def Normalization(self):
        '''
        There are a few methods for Normalization: STD, 0-1 and Special
        
        STD --> Normaliztion by STD
        0-1 --> Normaliztion between 0 to 1 by dividing in the columns maximum value
        Special --> Normalize by a value for wanted column average and wanted column STD
        
        Input:
        
            --------------------
            
            dataframe(=empty dataframe)
            col_for_method(={}) --> Need to be: {Method:[col1,col2]} --> The method can be: 'STD'/'0-1'/'Special'
            replace_columns(=True)
        
            --------------------
            
            <>If STD is in col_for_method keys you also need to provide the following arguments aswell:
            
              target(='')
              key(='')
              std_range(=[float('-inf'),0,1,float('inf')]) --> The groups are between each two values
              
            --------------------
            
            <>If Special is in col_for_method keys you need to provide the following arguments asweel:

              wanted_average(=0)
              wanted_std(=0)
              
            --------------------
            
        Output: 
        
            An updated dataframe
        '''
        
        if 'STD' in list(self.col_for_method.keys()):
            col_std = self.col_for_method['STD'] #Getting all the columns for STD method
            std_df = pd.DataFrame()
            #Getting for all our columns their normalize std values by: (value-column average)/column std
            std_obj = BI.Evaluation.Model_Evaluation(dataframe=self.dataframe, 
                         target=self.target, 
                         key=self.key, 
                         feats_for_std=col_std,
                         std_range=self.std_range,
                         std_range_labels=list(range(1,len(self.std_range))))            
            std_df = std_obj.STD()
            for col in col_std:
                col_df = pd.DataFrame()
                col_df = std_df[std_df['Feature']==col] #From the std dataframe we got, we keep only our current column name          
                self.dataframe = self.dataframe.merge(col_df[[self.key,'std']],on=self.key,how='left') #We merge our dataframes by the key to get the std
                
                if self.replace_columns == 1: #If we want to replace the original column
                    del self.dataframe[col] #Delete the originall column
                    self.dataframe = self.dataframe.rename(columns={'std':col}) #Change the std column to the originall columns name
                else: #If we dont want to replace the original column
                    self.dataframe = self.dataframe.rename(columns={'std':(str(col)+'_std')}) #Changing the std column name to something connected to the originall column
                
        if '0-1' in list(self.col_for_method.keys()):
            col_01= self.col_for_method['0-1'] #Getting all the columns for 0-1 method
            for col in col_01:
                max_value = max(self.dataframe[col]) #Max value in this column
                if self.replace_columns == 1: #If we want to replace the original column
                    self.dataframe[col] = self.dataframe[col]/max_value #Each value gets replace by - the value/the max value
                else: #If we dont want to replace the original column
                    self.dataframe[str(col)+'_01'] = self.dataframe[col]/max_value #Creats a new column for our calculation
                
        if 'Special' in list(self.col_for_method.keys()):
            col_nd= self.col_for_method['Special'] #Getting all the columns for Special method
            for col in col_nd:
                if self.replace_columns == 1: #If we want to replace the original column
                    #We calculate our normalize std: (value-column average)/column std
                    #We take our normalize std and: norm_std*wanted_std_wanted average
                    self.dataframe[col] = ((self.dataframe[col]-self.dataframe[col].mean())/self.dataframe[col].std())*self.wanted_std+self.wanted_average
                else: #If we dont want to replace the original column
                    #Creats a new column for our calculation
                    self.dataframe[str(col)+'_Special'] = ((self.dataframe[col]-self.dataframe[col].mean())/self.dataframe[col].std())*self.wanted_std+self.wanted_average

        return self.dataframe
    
    def WTF(self):
        '''
        Just bringing us the WTF data in convenient way, not really a replace column function but its the closest
        
        Input: 
        
            --------------------
            
            dataframe(=empty dataframe)
            key('')
            wtf_prefix(='WTF')
            outputpath(='') --> If left as '' than no file will be created
            add_boolwtf(=False)
            min_wtf_target_size(=1.0) If an int the code will look at it as a size, if a float it will look at it as a percent
            influencing_wtfs(=3) Number of the most influencing wtf we want(the best 3/4/10...)
            
            --------------------
               
        Output: 
        
            1) An updated dataframe with bool columns for the most influencing WTF's
            2) A dataframe with a key column and a WTF column - the same key 
        '''
        
        df_wtf = pd.DataFrame() #A dataframe for our results
        wtf_columns = [s for s in self.dataframe.columns.values if self.WTF_prefix in s] #Getting all the columns that start with the WTF prefix     
        for wtf in wtf_columns: #For each wtf column
            df_wtf_specific = self.dataframe[[self.key,wtf]] #Creating a dataframe with only our key and specific wtf
            df_wtf_specific = df_wtf_specific[df_wtf_specific[wtf]!=0].rename(columns={wtf:'WTF'}) #Slicing so we wont have a wtf=0, and also renaming the column to WTF
            df_wtf = df_wtf.append(df_wtf_specific,ignore_index=True) #Adding the specific dataframe to our main wtf dataframe
            
        df_wtf.sort_values(self.key,ascending=True,inplace=True) #Making it so the same key is in successive rows
        
        if self.outputpath!='':
            df_wtf.to_csv(self.outputpath,index=False,encoding='cp1255') #Export to csv
            
        #Only if want to add to our dataframe columns of the most influencing WTF's
        if self.add_boolwtf == True:
            df_wtf_for_full = df_wtf.merge(self.dataframe[[self.key,self.target]],on=self.key,how='left') #Adding to out wtf dataframe the target column
            if type(self.min_wtf_target_size)==int: 
                min_size = self.min_wtf_target_size #If it's an int we will just take the value
            elif type(self.min_wtf_target_size)==float:
                min_size = self.dataframe.shape[0]/(100/self.min_wtf_target_size) #If it's a float we see the value as a percent and the size will be that percent from the whole dataframe
                
            wtf_list = list(df_wtf_for_full['WTF'].unique()) #List of all our WTF's
            
            milon = {} #Holds the target rate values for each WTF
            milon2 = {} #Holds the size for each WTF
            for wtf in wtf_list:
                milon[wtf] = df_wtf_for_full[(df_wtf_for_full['WTF']==wtf)&(df_wtf_for_full[self.target]==1)].shape[0]/df_wtf_for_full[df_wtf_for_full['WTF']==wtf].shape[0]
                milon2[wtf] = df_wtf_for_full[df_wtf_for_full['WTF']==wtf].shape[0]
                
            df_wtf_data = pd.DataFrame() #A small dataframe with all the WTF's and their size and target rate
            df_wtf_data['WTF'] = wtf_list
            df_wtf_data['WTF_target_rate'] = df_wtf_data['WTF'].map(milon)
            df_wtf_data['WTF_target_shape'] = df_wtf_data['WTF'].map(milon2)
            
            df_wtf_data = df_wtf_data[df_wtf_data['WTF_target_shape']>=min_size] #We want only WTF's with a size bigger than the minimum size
            df_wtf_data.sort_values('WTF_target_rate',ascending=False,inplace=True) #Sort it from the biggest target rate to the smallest
            selected_wtfs = list(df_wtf_data['WTF'].head(self.influencing_wtfs)) #Take only the best n amount of WTF's
            
            for sel_wtf in selected_wtfs: #For each of the most influencing WTF's we chose
                sel_wtf_ppl = list(df_wtf_for_full[df_wtf_for_full['WTF']==sel_wtf][self.key]) #Only the people who have the specific WTF
                self.dataframe[sel_wtf] = np.where(self.dataframe[self.key].isin(sel_wtf_ppl),1,0) #For all the people - a bool colum indicating if they have this specific WTF or not
            
        return self.dataframe, df_wtf #Returns first an updated dataframe and secondly a dataframe of people and their WTF's

        
        
    def Fill_Empty_Values(self):
        '''Oh no! It seems like we are missing another explanation, 
           Haviv is going to be mad...'''
        pass  