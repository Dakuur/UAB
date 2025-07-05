import matplotlib as plt

# Draw the k-nn graph from the adjacency matrix
plt.clf()
G=nx.from_numpy_matrix(adj)
cols.append('black')
nx.draw(G,pos= X_clas,node_color = cols)
for i in range(n_examples_plot-1):
    txt = X_train_labels[i]
    plt.annotate(txt, (X_clas[i,0],X_clas[i,1]))
plt.annotate(X_test_labels[test_choice],(X_clas[-1,0],X_clas[-1,1]))
plt.show()

# Assign the class to the new example
index_clas=adj[-1:,0:n_objects_train].astype(bool)
index_clas=index_clas[0,]

