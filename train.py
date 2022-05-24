import pdb
import pickle
import sklearn
import xgboost as xgb
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
import joblib



def train(clf, training_data, weights, target):
    print(clf)

    sumWeightsSignal = np.sum(weights * (target>0))
    sumWeightsBackground = sum(weights * (target == 0))

    print('Sum weights signal', sumWeightsSignal)
    print('Sum weights background', sumWeightsBackground)

    aveWeightSignal = sumWeightsSignal/np.sum(target)
    print('Average weight signal', aveWeightSignal)
    aveWeightBG = sumWeightsSignal/np.sum(1-target)
    print('Average weight background', aveWeightBG)

    
    nCrossVal = 2
    kf = KFold(nCrossVal, shuffle=True, random_state=1)

    print('Cross-validation:', nCrossVal, 'folds')

    for trainIndices, testIndices in kf.split(training_data):
        print('Starting fold')

        d_train = training_data[trainIndices]
        d_test = training_data[testIndices]

        t_train = target[trainIndices]
        t_test = target[testIndices]

        w_train = weights[trainIndices]
        w_test = weights[testIndices]

        # del training_data, target, weights, trainIndices, testIndices, kf

        # import pdb; pdb.set_trace()

        clf.fit(d_train, t_train) #, sample_weight=w_train)

        print('Produce scores')
        scores = clf.predict_proba(d_test)
        scores = scores[:,1]

        fpr, tpr, thresholds = roc_curve(t_test, scores, sample_weight=w_test)

        joblib.dump((fpr, tpr, thresholds), 'roc_vals.pkl')

        effs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for eff in effs:
            print('Fake rate at signal eff', eff, fpr[np.argmax(tpr>eff)])

        break
    
    # Can save with different features if necessary
    joblib.dump(clf, 'train/{name}_clf.pkl'.format(name=clf.__class__.__name__), compress=9)

    # if doCrossVal:
    print('Feature importances:')
    print(clf.feature_importances_)

    # varList = trainVars()
    # for i, imp in enumerate(clf.feature_importances_):
    #     print(imp, varList[i] if i<len(varList) else 'N/A')
    
    return clf


def readFiles():
    print('Reading files...')

    coffea_out = pickle.load(open('result_220524_ML_emutau_v1.pkl', 'rb'))

    arr = coffea_out['sel_array'].value

    targets = arr[:,0] # 0 for BG, HNL mass for HNLs
    weights = arr[:,1]
    arr_vars = arr[:, 2:]

    targets = (targets > 0).astype(int) # just 1 for signal
    weights = weights * (weights > 0)

    print('Done reading files.')

    return arr_vars, weights, targets

if __name__ == '__main__':

    print('Read training and test files...')
    training, weights, targets = readFiles()        

    print('Sizes')
    print(training.nbytes, weights.nbytes, targets.nbytes)

    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, eta=0.05, min_child_weight=0., subsample=0.7, seed=1234)
    train(clf, training, weights, targets)


    