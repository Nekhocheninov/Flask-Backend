from sklearn.linear_model import LogisticRegression
from sklearn import metrics

trained_model = None

def train_model(train_data):
    
    train_data['Cabin'] = train_data['Cabin'].fillna('N').str.get(0).replace(['N','A','B','C','D','E','F','G','T'], [0,1,2,3,4,5,6,7,8])
    train_data['Embarked'] = train_data['Embarked'].replace(['C','Q','S'], [0,1,2])

    import re
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""

    train_data['Title'] = train_data['Name'].apply(get_title)
    train_data['Title'] = train_data['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    train_data['Title'] = train_data['Title'].replace(['Mlle','Ms','Mme'], ['Miss','Miss','Mrs'])
    train_data['Title'] = train_data['Title'].replace(['Mr','Mrs','Miss','Master','Rare'], [0,1,2,3,4])

    X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Title']].replace(['female', 'male'], [1, 2]).values.reshape(-1, 8)
    Y = train_data['Survived'].values.reshape(-1, 1).ravel() #.replace([0, 1], [1, 0])

    logreg = LogisticRegression(max_iter=2000).fit(X, Y)

    global trained_model
    trained_model = {'logreg': logreg, 'X': X, 'Y': Y}

    return {'logreg': logreg, 'X': X, 'Y': Y}

def predict():
    logreg = trained_model['logreg']
    X = trained_model['X']
    Y = trained_model['Y']

    fpr, tpr, thresholds = metrics.roc_curve(Y, logreg.predict_proba(X)[:, 1])
    Y_p = logreg.predict(X)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'X': X,
        'Y': Y,
        'Y_p': Y_p}
