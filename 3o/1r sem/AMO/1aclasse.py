te voy a pasar preguntas tipo test. con este archivo responde brevemente solo con la opcion correcta




Volem implementar un KNN amb k= 9 amb K-Fold Cross Validation de 10 particions. Després, volem saber la accuracy representativa dels 10 models generats amb el CV. Tria l'opció correcta.

kf = KFold(n_splits=10, shuffle=True, random_state=42)

k_value = 9

Pregunta 1Resposta

a. model = KNeighborsClassifier(n_neighbors=k_value) scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy') accuracy = scores.max()

b. model = KNeighborsClassifier(n_neighbors=kf) scores = cross_val_score(model, X, y, cv=k_value, scoring='accuracy') accuracy = scores.max()

c. model = KNeighborsClassifier(n_neighbors=k_value) scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy') accuracy = scores.mean()

d. model = KNeighborsClassifier(n_neighbors=k_value)

scores = cross_val_score(model, X, y, cv=kf, scoring='precision_macro')

accuracy = scores.mean()





Quina és l'assignació correcta per la función roc_curve de sklearn.metrics?

Pregunta 2Resposta

a. tpr, thresholds, fpr = roc_curve(y_pred_prob[:,1], y_test_pima)

b. fpr, tpr, thresholds = roc_curve(y_test_pima, y_pred_prob[:,1])

c. fpr, tpr, thresholds = roc_curve(y_pred_prob, y_test_pima)

d. thresholds, fpr, tpr = roc_curve(y_pred_prob[:,1], y_test_pima)






A quina mètrica de rendiment correspon aquesta funció?

    def ???(self):
        return (self._tp + self._tn)/(self._tp + self._tn + self._fp + self._fn)
   

Pregunta 3Resposta

a. Accuracy

b. Recall

c. Precision

d. F1 Score
