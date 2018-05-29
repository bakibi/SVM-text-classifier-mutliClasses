from sklearn.model_selection import train_test_split
try:
    # Fix UTF8 output issues on Windows console.
    # Does nothing if package is not installed
    from win_unicode_console import enable
    enable()
except ImportError:
    pass
import re
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




'''
    Enlever d'une chaine de caractere tous les caracteres qui peuvent alter le
        jugement ou la transformation sur le resultat .
'''
def clean_str(s):
	s = re.sub(r"[^\u0627-\u064aA-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

'''
    Lire un fichier qui contient un axe de tete , sous la forme suivantes :
        ['Category', 'Descript'] ,et ensuite retourner ces valeurs desordonnees.
'''

def read_file(nom_fichier,ziper=True):
    if ziper ==True:
        values = pd.read_csv(nom_fichier, compression='zip', encoding='utf-8')
    else:
        values = pd.read_csv(nom_fichier, encoding='utf-8')
    selected = ['Category', 'Descript']
    non_selected = list(set(values.columns) - set(selected)) # supprimer les colonne qu on a pas besoin
    values = values.drop(non_selected, axis=1)
    values = values.dropna(axis=0, how='any', subset=selected)
    values = values.reindex(np.random.permutation(values.index))
    return values

'''
    Tranformer des donnnes 'values' en des donnnees qu on peut traiter ,
        cv dire les labels en les transforme sous forme d un vecteur qui a
        que des 0 et 1 ,et aussi en separrant les features
'''

def traiter_donnees(values,selected=['Category', 'Descript']):
    # extraire les categories
    labels = sorted(list(set(values[selected[0]].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int) #matrice carree
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    op1 =  values[selected[0]]
    labels = [label_dict[x] for x in op1]
    # les description en une liste
    features = values[selected[1]]
    return features,labels


'''
    Cette fonction permet de transformer les donnees features alpha --> num
        et le vector doit avoir une taille y qui est egale a MAX_SEQUENCE
        aussi cette fonction retourne le dictionnnire de donnnes , qui a
        servit a la transformation des donnnees .
'''

def transformer_features(features,MAX_SEQUENCE=30,est=False,dict1=None):
    # je decompose les description en une liste de listes ou le separator est les espaces
    print('decomposition')
    features = features.apply(lambda x: clean_str(x).split(' ')).tolist()
    #faire un groupe de mots


    if est == False:
        print('making a set')
        sett = set()
        llist = []
        for i in range(len(features)):
            for j in range(len(features[i])):
                sett.add(features[i][j])
        # faire un dictonnaire
        print('faire un dictionnaire')
        dict1 = {}
        total = len(sett)
        i = 1
        for a in sett:
            dict1[a] = (i)*1.0/total
            i = i+1
    else:
        total = len(dict1)
    # reformer les donner
    features = [[dict1[x1] for x1 in x] for x in features]
    new_feature = []

    for i in range(len(features)):
        x = len(features[i])
        e = []
        if x <= MAX_SEQUENCE :
            e = [a  for a in features[i]]
            e = e + [0.1/total]*(MAX_SEQUENCE - x)
        else:
            e  = features[i][0:MAX_SEQUENCE]
        new_feature.append(e)
    return new_feature,dict1


def labels_choose(features,labels,ind1,ind2):
    features_ = []
    labels_ = []
    for i in range(len(features)):
        if np.argmax(labels[i]) == ind1:
            features_.append(features[i])
            labels_.append(1)
        if np.argmax(labels[i]) == ind2:
            features_.append(features[i])
            labels_.append(-1)
    return features_,labels_

'''
    return the accuary of a set of output

'''
def accuracy_output(real_output,predicted_output):
    ans = 0.0
    tout  = len(real_output)
    for i in range(tout):
        if real_output[i] == predicted_output[i]:
            ans = ans + 1.0

    ans = ans / tout
    return ans

'''
x,y =traiter_donnees(read_file('data/train1.csv.zip'))
transformer_features(x)
'''
