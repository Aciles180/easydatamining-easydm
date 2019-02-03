# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:31:34 2017

@author: s8350866
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from scipy import stats
#from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from itertools import combinations

class Data_Analyze(object):
    '''
    This class as function to help you analyze your final data

    General:
    
    outputpath:                                                                default=''
    dataframe:                                                                 default=pd.DataFrame() 
    target:                                                                    default=''
    
    Decision_trees:
    
    features:                                                                  default=[]
    for_column:                                                                default=''
    
    Graphs:
    
    graph_features:                                                            default=[]
    
    Group_Comparison:
    
    key:                                                                       default=''
    compare_method: 'Values'/'Distribution'                                    default='Value'
    compare_col:                                                               default=''
    distribution_col:                                                          default=''  
    special_test_outputpath:                                                   default=''
    '''
    
    def __init__(self, 
                 outputpath='',
                 dataframe=pd.DataFrame(), 
                 key='', 
                 target='',
                 features=[], 
                 for_column='', 
                 graph_features=[],
                 compare_method='Value',
                 compare_col='',
                 distribution_col='',
                 special_test_outputpath=''):
        
        #General
        self.outputpath = outputpath
        self.dataframe = dataframe
        self.target = target
        
        #Decision trees           
        self.features = features
        self.for_column = for_column
        self.feature_name = []
        self.tree = ()
        self.df_res = pd.DataFrame()
        
        #Graphs
        self.graph_features = graph_features
        
        #Comparing
        self.key = key
        self.compare_method = compare_method
        self.compare_col = compare_col
        self.distribution_col = distribution_col
        self.special_test_outputpath = special_test_outputpath
        

    def recruse(self,node,depth,rule,for_col_val):
        if self.tree.feature[node]!=_tree.TREE_UNDEFINED: #checking if the node given is a index in our feature in trees indexes
            name=self.feature_name[node] #The name of the feature
            threshold=self.tree.threshold[node] #The treshold for the feature
         
            left=self.tree.children_left[node] #The second branch of the left side
            right=self.tree.children_right[node] #The second branch of the right side
            rule1=rule+" "+name+"<="+str(threshold) #Creating the rule for the left side as it will always be <
            rule2=rule+" "+name+">="+str(threshold) #Creating the rule for the left side as it will always be >
            
            #Running this function again for each secondary branch (left and right)
            #Our node is the index for the feature inside the secondaary branch
            #The depth increases to show us on wich secondary branch we are(it can get to as many features we have)
            #for_col_val is the specific value in the for column we are on, as we dont want to mix them
            Data_Analyze.recruse(self,self.tree.children_left[node],depth+1,rule1,for_col_val) 
            Data_Analyze.recruse(self,self.tree.children_right[node],depth+1,rule2,for_col_val)
            if for_col_val!='': #If we do a loop on column
                record={self.for_column:for_col_val,'Pattern1':rule1,'Pattern1_size':self.tree.n_node_samples[left],'Pattern1_nesher':self.tree.value[left][0][1]/self.tree.n_node_samples[left],'Pattern2':rule2,'Pattern2_size':self.tree.n_node_samples[right],'Pattern2_nesher':self.tree.value[right][0][1]/self.tree.n_node_samples[right]}
            else:
                record={'Pattern1':rule1,'Pattern1_size':self.tree.n_node_samples[left],'Pattern1_nesher':self.tree.value[left][0][1]/self.tree.n_node_samples[left],'Pattern2':rule2,'Pattern2_size':self.tree.n_node_samples[right],'Pattern2_nesher':self.tree.value[right][0][1]/self.tree.n_node_samples[right]}
            self.df_res=self.df_res.append(record,ignore_index=True) #Adding the rule to our specific dataframe
             
    def Decision_trees(self):
        '''
        Creats a dicision tree file
        
        Input: 

            --------------------        
            
            dataframe(=empty dataframe)
            target(='')
            features(=[])
            for_column(='')
            outputpath(='') 
            
            --------------------
               
        Output: 
        
            A CSV file with 6/7 columns: If we do the trees on one values - Pettern1, Pattern1 target rate, pattern1 size
            and the same for Pattern2. If we do the trees for each unique value in a column than we have another column 
            to indicate for which value is the pattern right
        '''
        
        df_all = pd.DataFrame() #The final dataframe: What we do for on(optional),pattern1,Pattern1_nesher,Pattern1_size,Pattern2,Pattern2_nesher,Pattern2_size  
        estimator=DecisionTreeClassifier(max_leaf_nodes=5,min_samples_leaf=50,max_depth=5,random_state=0) #Creating the object so we can use the DecisionTreeClassifier function
        
        #For each feature (only integers!) we replace the empty values with the column average
        for feat in self.features:
            x = self.dataframe[feat].mean()
            self.dataframe[feat] = self.dataframe[feat].replace(np.NaN,x)
        
        if self.for_column != '': #If we have a column that we want to do the trees on each of its unique values
            for a in self.dataframe[self.for_column].unique():
                df2 = self.dataframe[self.dataframe[self.for_column]==a] #Creating a dataframe whith only the value we are on
                estimator.fit(df2[self.features],df2[self.target]) #Fitting our DecisionTreeClassifier object to our dataframe
                
                self.tree = estimator.tree_ #Taking our tree object
                #For each feature index in self.tree.feature, if we use it in a tree we add its name to the list, if not we add UND to the list
                self.feature_name = [self.features[i] if i!=_tree.TREE_UNDEFINED else "UND" for i in self.tree.feature]              
                self.df_res = pd.DataFrame() #Creating a new dataframe for the specific column value we are on
                Data_Analyze.recruse(self,0,0,"",a) #Using another function - explenation there
                
                df_all = df_all.append(self.df_res) #Adding the specific dataframe to our full dataframe

        else:
            estimator.fit(self.dataframe[self.features],self.dataframe[self.target])
            
            self.tree = estimator.tree_ #Taking our tree object
            #For each feature index in self.tree.feature, if we use it in a tree we add its name to the list, if not we add UND to the list
            self.feature_name = [self.features[i] if i!=_tree.TREE_UNDEFINED else "UND" for i in self.tree.feature]
            
            self.df_res = pd.DataFrame() #Creating a new dataframe for the specific column value we are on
            Data_Analyze.recruse(self,0,0,"",'') #Using another function - explenation there
            
            df_all = df_all.append(self.df_res)
            
        df_all.to_csv(self.outputpath,index=False,encoding='cp1255') #Exporting to a csv
    
    def Graphs(self):
        '''
        Surprisingly it creates graphs...
        
        Input: 
        
            --------------------
            
            dataframe(=empty dataframe)
            target(='') 
            graph_features(=[])
            outputpath(='') --> Need to be a folder and not a file name
            
            --------------------
            
        Output: 
        
            A jpg file for each graph
        '''
        
        for feat in self.graph_features: #A loop on all the features we want to do graphs on
            df_sample = self.dataframe[[feat,self.target]] #Creating a dataframe with only our feature column and the target column
            pivot = df_sample.pivot_table(index=feat, values=self.target,aggfunc=[np.mean,np.size]).reset_index() #We are creating a pivot that return our target sum and average
            pivot=pivot.set_index(feat) #We are making our feature column the index so we can transfer our dataframe to dictionary easly
            pivot1=pivot.to_dict() #Transfering our dataframe to a dictionary
            
            ind=np.arange(pivot.shape[0]) #Getting an arange of the amount of unique values
            fig,ax=plt.subplots() #Creating the graph canvas(we will need to add the data)
                
            r=ax.bar(ind,pivot['mean']*100,0.35,align='center',color='red') #Creating the bars in our graph
            plt.ylim(0,100) #Setting our Y scale to averages so its between 0-100
            plt.xlim(-1,len(pivot['mean'])) #Setting our X scale to the number of unique values
            ax.set_xticks(np.arange(len(pivot1['size']))) #Setting the correct position for the X labels by the amount of values
            ax.set_xticklabels(labels=pivot.index.values,rotation=45,fontname='Comic Sans MS') #Setting the X labels for the columns
            lst=list(pivot.index.values) #The unique values for this feature
            i=0
            for bar in r: #For each of the bars in our graph
                height=bar.get_height() #We get the bar height
                #Down here we add the Y labels for our bars, and modify them as we want
                ax.text(bar.get_x()+bar.get_width()/2.,height,'%s'%(str(int(height))+'% '+str(pivot1['size'][lst[i]])),ha='center',va='bottom',fontname='Arial',fontsize=14)
                i=i+1

            fig.savefig(self.outputpath + str(feat) + '.jpg') #We save our graph as a jpg 
            
    def Group_Comparison(self):
        '''
        This function is used to compare between the different values of a column, for example
        for each unit or profession. There are two ways to compare between groups: Value and Distribution
        
        Value --> If we only care for the mean of the column
        Distribution --> If we only want to check how the people in each group distribute between the values of another column

        Input: 
        
            --------------------
            
            <>For both comparing methods:
            
            dataframe(=empty dataframe)
            target(='') 
            compare_col(='')    
            compare_method(='Value') --> Can be: 'Value'/'Distribution'
            outputpath(='')
            special_test_outputpath(='')
            
            --------------------
            
            <>If compare_method is 'Distribution' we also need the following argument:
            
            distribution_col(='')
            key(='')

        Output: 
        
            <>If compare_method is 'Value':
                
              A gradual  comparing file by the values and a T-test file.
              
            <>If compare_method is 'Distribution':
            
              A file with a pivot in which the columns are the equated values and the rows are the comparing column values, 
              we also get a file with chisquare test        
        '''


        df_full = pd.DataFrame()
        if self.compare_method == 'Value': #If we only care for the mean of the column
            #1 - Pivot for each value we compare we show there target average
            table = pd.pivot_table(self.dataframe,columns=[self.compare_col],values=self.target,aggfunc=np.mean).reset_index()
            
            #2 - We are doing a T-test for each combination of our compared values 
            unique_vals = self.dataframe[self.compare_col].unique() #Each unique value we compare
            combs=list(combinations(unique_vals,2)) #All the combination of two compared values
            for comb in combs: #For each combination
                x = self.dataframe[self.dataframe[self.compare_col]==comb[0]] #The dataframe with only our first compared value
                x = x[self.target] #Keeping only the target column
                y = self.dataframe[self.dataframe[self.compare_col]==comb[1]] #The dataframe with only our second compared value
                y = y[self.target] #Keeping only the target column
                t_result = stats.ttest_ind(x,y,equal_var=False) #T-test
                pvalue = t_result[1] #We take only the second result which is the Pvalue
                row = {'First Value':comb[0],'Second Value':comb[1],'Ttest_Pvalue':pvalue} #Summary of this comparing
                df_full = df_full.append(row,ignore_index=True) #Adding it to our full dataframe
                
        if self.compare_method == 'Distribution': #If we onlywant to check how the people in each group distribute between the values of another column
            #3 - Pivot for each compared value(rows) we check its ditribution for each compared target value(column) by counting the number of keys
            table = pd.pivot_table(self.dataframe,columns=[self.distribution_col],index=[self.compare_col],values=self.key,aggfunc=np.size).reset_index()
            
            #4 - We are doing Chisquare-test for each of our compared values distribution of keys
            uniq_comp_vals = self.dataframe[self.compare_col].unique() #Each unique value we compare 
            uniq_tar_vals = self.dataframe[self.distribution_col].unique() ##Each unique value of the target
            comp_dict = {} #Full dict with the list of distributions
            for comp_val in uniq_comp_vals: #For each compared value
                comp_dict[comp_val] = [] #We add it is a key to our dictionary
                df_comp_spec = self.dataframe[self.dataframe[self.compare_col]==comp_val] #A dataframe with only our the specific compared value
                uniq_tar_vals_specific = df_comp_spec[self.distribution_col].unique() #Every target value for our specific compared value
                count = 0 #For counting the number of keys
                for tar_val in uniq_tar_vals: #For each target value, so it will always be in the same order
                    if tar_val in uniq_tar_vals_specific: #If we got this value in the list of values for our specific dataframe
                        count = df_comp_spec[df_comp_spec[self.distribution_col]==tar_val].shape[0] #We count the keys
                    else:
                        count = 0 #If its not in the list it means we dont have any keys with it so its 0
                    comp_dict[comp_val].append(count) #Adding the count to our list of counts(=the distribution)
                    
            combs=list(combinations(uniq_comp_vals,2)) #All the combination of two compared values
            for comb in combs: #For each combination
                li = comp_dict[comb[0]].copy() #Getting our distribution list for the first compared value
                li2 = comp_dict[comb[1]].copy() #Getting our distribution list for the second compared value
                check = [x+y for x,y in zip(li,li2)] #Making a list of the summs of the two lists(by the index)
                check2 =[i for i,val in enumerate(check) if val==0] #We are getting the list of all the indexes were the summ is 0, which means that in the two compared values there are 0 keys
                for ind in sorted(check2,reverse=True): #For each index that the sum is a zero
                    del li[ind] #We delete the value from this index in the first distribution list
                    del li2[ind] #We delete the value from this index in the second distribution list

                #obs = np.array([li,li2])
                chi_test = chi2_contingency([li,li2]) #Chisquare-test
                pvalue = chi_test[1] #We take only the second result which is the Pvalue
                row = {'First Value':comb[0],'Second Value':comb[1],'Chisquare_Pvalue':pvalue}#Summary of this comparing
                df_full = df_full.append(row,ignore_index=True) #Adding it to our full dataframe
                 
        table.to_csv(self.outputpath,index=False,encoding='cp1255') #Exporting our regular pivot to a csv file
        df_full.to_csv(self.special_test_outputpath,index=False,encoding='cp1255') #Exporting our special pivot to a csv file
        return table,df_full
    