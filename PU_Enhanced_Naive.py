### Loading packages and auxiliary functions
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_breast_cancer # for exemplary data set

def sigma(s):
    '''
    Logistic function
    '''
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Turns off redundant warnings
        # about overflow in exp(s)

        result = np.where(
            s > 0,
            1 / (1 + np.exp(-s)),
            np.exp(s) / (np.exp(s) + 1) 
            # Otherwise for positive s we could get quotient of two infinities
        )

    return result


def create_s(y, c):   # c - labeling frequency
    '''
    Creates array s from a given target variable y and value of c
    '''
    s = np.array(y)
    positives = np.where(y == 1)[0]

    new_unlabeled_samples = positives[np.random.random(len(positives)) < 1 - c]
    s[new_unlabeled_samples] = 0
    
    return s


### Enhanced classifier
class Enhanced:
    
    def __init__(self):

        self.interc = 0
        
    def fit(self, X, s):

        # First step: fitting naive model
        mod_log_naiw = LogisticRegression( penalty = 'none', max_iter = 3000 )
        self.coef = mod_log_naiw.fit(X, s).coef_  #array with beta_{-0}
        
        # Second step: calculating intercept
        product = np.matmul(X, self.coef.reshape(X.shape[1], ))
        arg_sort = np.argsort(-product)
        prod_sorted = -product[arg_sort]
        s_sorted = s[arg_sort]
        
        boundary_min = prod_sorted >= -max(product)
        ind_start = np.argmax(boundary_min > 0.5)  # The smallest index for
        # which there are any predictions belonging to the positive class
               
        ind1 = (s_sorted == 1)
        sum_ind1 = sum(ind1)
        n = len(s_sorted)
        
        j = 0
        arg = prod_sorted[ind_start: ]
        val = list(np.zeros( X.shape[0] - ind_start ))
        numerator = 0 
        if s_sorted[ind_start] == 1:
            numerator = 1   
            
        denominator = 1
        val[0] = ( (numerator / sum_ind1 )**2 ) / ( 1 / n )  
        
        # Calculating values of F1 corresponding to all considered
        # values of intercept.
        # Arbitrarly, positive class is assigned also to observations with 
        # a posteriori probability equal to 0.5.
        for i in np.arange(ind_start + 1, X.shape[0] - 1):
            j = j + 1
            if s_sorted[i] == 1:
                numerator = numerator + 1

            denominator = denominator + 1
            
            if ( (prod_sorted[i] != prod_sorted[i + 1]) ):
                val[j] = ( (numerator / sum_ind1 )**2 ) / ( denominator / n )
            else:
                val[j] = val[j - 1]
            
        arg_max = np.argmax( val )
        
        self.interc = arg[arg_max]
        
    def predict_proba(self, X):
        """
        Evaluates a posteriori probabilities.
        """
        y_pred = np.array( sigma( np.matmul(X, self.coef.reshape(X.shape[1], )) + self.interc ) )
        y_pred = y_pred.reshape(-1, 1)
        return np.hstack( (1 - y_pred, y_pred) )
    
    def predict(self, X):
        """
        Assigns classes corresponding to the highest a posteriori
        probability.
        """
        pred_proba = self.predict_proba( X )

        return ( pred_proba[:, 1] > 0.5 ).astype(int)

    
### Loading exemplary data set
data = load_breast_cancer( return_X_y = True, as_frame = True )
# X.head()
X = np.array( data[0].iloc[:, 0:5] )
y = np.array( data[1] )


### Defining test as a function
def test(X, y, c = 0.5, test_size = 0.2, n_best_features = 5):
    s = create_s(y, c)
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size = test_size)

    mod_Enhanced = Enhanced( )
    mod_Enhanced.fit(X_train, s_train)
    print('Estimated parameters:')
    print(str.format('intercept = {0:.4f}, ', mod_Enhanced.interc))
    print('beta_{-0} = ', mod_Enhanced.coef)
    print('')


    pred_f1 = mod_Enhanced.predict( X_test )

    print( str.format("F1 measure on a test set: {0:.4f}", 
                      metrics.f1_score( y_test, pred_f1 )) )
    print('')
    print( str.format("Balanced accuracy on a test set: {0:.4f}",
                      metrics.balanced_accuracy_score( y_test, pred_f1 )) )


### Running test
test(X, y)    
