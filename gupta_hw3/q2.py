"""
FileName: q2.py
Author: prakhar gupta pg9349
Description: Create a Naive Bayes classifier
"""


import  numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

class NB():
    """
    Naive Bayes CLassifier
    """
    def __init__(self, alpha=1, flag=1):
        self.alpha = alpha
        self.rows=0
        self.classes=None
        self.mean, self.var=[],[]
        self.count=None
        self.class_counts=None
        self.num_feature,self.cat_feature=[],[]
        self.prior=None

    def calculate_prior(self,features,target):
        '''
        calculate prior of the class with class counts
        '''
        self.classes_counts=features.groupby([target]).count()
        self.class_counts = self.classes_counts.iloc[:, 0]
        # print(self.classes_counts)
        l=features.shape[0]
        self.prior=self.class_counts/l

        # print(self.prior)


    def calc_statistics_num(self, features, target):
        '''
        calculate mean and variance for each numerical column and stores it in a pandas groupby object
        '''
        # print(features.columns)
        self.mean = features.groupby(target).mean()
        self.var = features.groupby(target).var()

        self.num_feature=self.num_feature+list(features.columns)
        return self.mean, self.var

    def calc_statistics_cat(self, features, target):
        '''
        calculate counts for each cat column and stores it in a pandas groupby object
        '''
        col=features.columns
        features=pd.get_dummies(features[col])

        self.count = features.groupby(target).sum()

        self.cat_feature=self.cat_feature+list(features.columns)
        return self.count


    def gaussian_density(self, class_idx, col1,x):
        '''
        calculate probability from gaussian density function (normally distributed)
        we will assume that probability of specific target value given specific class is normally distributed

        probability density function derived from wikipedia:
        (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
        '''
        mean = self.mean.at[class_idx,col1]
        var = self.var.at[class_idx,col1]
        numerator = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))

        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob

    def fit(self, data,target):
        """
        Trainer functions
        :param data: data frame
        :param target: column to train on
        :return: None
        """

        self.classes = np.unique(data[target])
        # print(self.classes)
        self.calculate_prior(data,target)
        self.feature_nums = data.shape[1]
        cat=data.select_dtypes(include=[object])
        numeric=data.select_dtypes(exclude=[object])
        cat[target]=data[target]
        numeric[target]=data[target]


        self.rows = data.shape[0]
        if len(numeric)>0:
            self.calc_statistics_num(numeric, target)
        if len(cat)>0:

            self.calc_statistics_cat(cat, target)






    def test(self, data):
        """
        This is the predict function
        @params: data - to classifier
        """
        prob=[]
        # cat=data.select_dtypes(include=[object])
        # numeric=data.select_dtypes(exclude=[object])
        cat=pd.get_dummies(data)
        # print(cat)
        col=cat.columns
        # print("________")
        # print(col)
        previousi=0
        for i,k in cat.iterrows():
            # print(i)
            elemprob = {}
            for j in k.index:

                # print("________")
                # print(self.classes)
                # print(self.cat_feature)
                if j in self.cat_feature:

                    if k[j]==1:

                        for c in self.classes:
                            prior=self.prior[c]
                            class_count=self.class_counts[c]
                            # print(j)
                            likelihood_count=self.count.at[c,j]
                            likelihood=(likelihood_count/class_count)
                            elemprob[c]=elemprob.get(c,1)*likelihood
                elif j in self.num_feature:
                    for c in self.classes:

                        likelihood = self.gaussian_density(c,j,k[j])
                        elemprob[c] = elemprob.get(c, 1) * likelihood
            for c in self.classes:
                prior = self.prior[c]
                elemprob[c] = elemprob.get(c, 1) * prior
            prob.append(elemprob)

        return prob




def fullset(df,allcol=['in html',' has emoji', ' sent to list', ' from .com', ' has my name',
           ' has sig', ' # sentences', ' # words']):
    """
    Trains the model on whole dataset
    :param df: datafram
    :param allcol: columns
    :return: classifier
    """
    n=NB()

    cat = df.select_dtypes(include=[bool])
    # print(pd.get_dummies(df))
    col=cat.columns
    # for i in df.itertuples():
    #    print(i)

    for i in col:
        if i !=' is spam':
            df[i] = df[i].map({True: 'True', False: 'False'})

    # allcol=['in html', ' has emoji', ' sent to list', ' from .com', ' has my name',
    #        ' has sig', ' # sentences', ' # words']

    allcol_plus_target=allcol+[' is spam']
    # print(df.columns)
    n.fit(df[allcol_plus_target],' is spam')

    # print(df.columns)
    x=n.test(df[allcol])
    pred=[]
    for i in x:
        pred.append(max(i, key=i.get))
    y3=np.array(pred)
    y=df[' is spam'].values
    accuracy = (y3 == y).sum() / y3.shape[0]
    print("Accuracy of training data")
    print(accuracy)
    return n

def classify(df,n,
    allcol=['in html',' has emoji', ' sent to list', ' from .com', ' has my name',
           ' has sig', ' # sentences', ' # words']):
    """
    CLassifier based on the trained classifier
    :param df: dataframe
    :param n:
    :param allcol:
    :return:
    """
    cat = df.select_dtypes(include=[bool])
    # print(pd.get_dummies(df))
    col=cat.columns
    # for i in df.itertuples():
    #    print(i)

    for i in col:
        if i !=' is spam':
            df[i] = df[i].map({True: 'True', False: 'False'})

    x=n.test(df[allcol])
    pred=[]
    for i in x:
        pred.append(max(i, key=i.get))
    y3=np.array(pred)
    y=df[' is spam'].values
    accuracy = (y3 == y).sum() / y3.shape[0]

    return y3,accuracy

from itertools import chain, combinations

def powerset(s,f):
    """
    Create subsets of the whole set s
    :param s: list of columns
    :param f: minimum number of columns needed
    :return:
    """
    return chain.from_iterable(combinations(s, r) for r in range(f,len(s)+1))

def subsettrain(df,number=7,allcol=['in html',' has emoji', ' sent to list', ' from .com', ' has my name',
           ' has sig', ' # sentences', ' # words']):
    """
    Run a search to find the optimal features by creating n NB models to compare accuracy
    :param df: dataframe
    :param number:  minimum number of feature to be there in a model
    :param allcol: all the columns
    :return: best accuracy,best feature list and the model
    """
    x=powerset(allcol,number)
    maxaccuracy=0
    bestcol=[]
    bestclassi=None
    for i in x:
        allcol=list(i)


        n=NB()

        cat = df.select_dtypes(include=[bool])
        # print(pd.get_dummies(df))
        col=cat.columns
        # for i in df.itertuples():
        #    print(i)

        for i in col:
            if i !=' is spam':
                df[i] = df[i].map({True: 'True', False: 'False'})

        # allcol=['in html', ' has emoji', ' sent to list', ' from .com', ' has my name',
        #        ' has sig', ' # sentences', ' # words']

        allcol_plus_target=allcol+[' is spam']
        # print(df.columns)
        n.fit(df[allcol_plus_target],' is spam')

        # print(df.columns)
        dfval = pd.read_csv("q3b.csv")
        pred,accuracy=classify(dfval,n,allcol)
        # x=n.test(df[allcol])
        # pred=[]
        # for i in x:
        #     pred.append(max(i, key=i.get))
        # y3=np.array(pred)
        # y=df[' is spam'].values
        # accuracy = (y3 == y).sum() / y3.shape[0]
        # print("Accuracy of training data")
        # print(accuracy)



        if accuracy>maxaccuracy:
            maxaccuracy=accuracy
            bestclassi=n
            bestcol=allcol

    return maxaccuracy,bestcol,bestclassi




def save_params(n):
    f = n.count
    x = n.class_counts
    # print(f)
    # print(x)
    likelihood = f.divide(x, axis='index')
    likelihood.to_csv("Cateorical_Maximum_likelihood_values.csv")
    mean = n.mean
    var = n.var
    mean.to_csv("Mean_numerical_features.csv")
    var.to_csv("Var_numerical_features.csv")

print("____________________________________________")
print("Whole data")
print("____________________________________________")
df=pd.read_csv("q3.csv")
n=fullset(df)
print("____________________________________________")
print("Validation data")
df=pd.read_csv("q3b.csv")
predictions,accuracy=classify(df,n)
print("Accuracy of New data using all columns")
print(accuracy)
print("Error")
print(1-accuracy)
save_params(n)
print("____________________________________________")
print("subset data")
print("____________________________________________")
df=pd.read_csv("q3.csv")
a,b,c=subsettrain(df,number=6)
print("Best Subset")
print(b)
print("Best Accuracy")
print(a)
print("Error")
print(1-a)
print("____________________________________________")
