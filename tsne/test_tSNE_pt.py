# testing code from https://github.com/KlugerLab/FIt-SNE/blob/master/examples/test.ipynb

import os
import numpy as np
import pylab as plt
import seaborn as sns; sns.set()


from fast_tsne import fast_tsne



if __name__ == "__main__":
    g_code_save_root = r"D:\3rdPartyLib\Git_Hub\tSNE\FIt-SNE\dataset\car_spl_code1874"
    print("data using:", g_code_save_root)
    code_name_Lst = os.listdir(g_code_save_root)

    X = y = None
    instance_num_1pers = 0
    print("cur order:", code_name_Lst)
    for indx, code_name_i in enumerate(code_name_Lst):
        code_i_Arr = np.load(os.path.join(g_code_save_root, code_name_i)).astype('float64')
        code_i_Arr = code_i_Arr.reshape(code_i_Arr.shape[0], code_i_Arr.size//code_i_Arr.shape[0])
        label_i_Arr = np.full(code_i_Arr.shape[0], indx).astype('int')
        # print(code_i_Arr.shape)
        if indx == 0:
            X = code_i_Arr
            y = label_i_Arr
            instance_num_1pers = code_i_Arr.shape[0]
        else:
            X = np.concatenate([X, code_i_Arr])
            y = np.concatenate([y, label_i_Arr])
    print(X.shape)
    print(y.shape)



    pca_dim_use = 50
    # Do PCA and keep 50 dimensions
    X = X - X.mean(axis=0)  # (70000, 784)
    U, s, V = np.linalg.svd(X, full_matrices=False) # 
    X50 = np.dot(U, np.diag(s))[:,:pca_dim_use]

    # We will use PCA initialization later on
    PCAinit = X50[:,:2] / np.std(X50[:,0]) * 0.0001

    # 10 nice colors
    col = np.array(['#a6cee3', # light blue
                    '#1f78b4', # blue
                    '#b2df8a', # lighrt green
                    '#33a02c', # green 
                    '#fb9a99', # light red
                    '#e31a1c', # red
                    '#fdbf6f', # light orange
                    '#ff7f00', # orange
                    '#cab2d6', # light purple
                    '#6a3d9a', # purple
                    '#f59adb', # light pink 
                    '#db1da5', # pink 
                    '#9af5ed', # light cyan
                    '#09e8d4', # cyan
                    ])


    # Subsampling 
    # np.random.seed(42)
    # ind1k = np.random.choice(X.shape[0], 1000, replace=False)
    # ind5k = np.random.choice(X.shape[0], 5000, replace=False)

    Z = fast_tsne(X50, perplexity= pca_dim_use , seed=42, 
                learning_rate=3000,
                # stop_early_exag_iter=500, # default_lr = 200
                initialization=PCAinit ,
                ) 
    # np.save(r"C:\Users\admin\Desktop\Z_tsne_buffer.npy", Z)
    plt.figure(figsize=(7,7))
    plt.scatter(Z[:,0], Z[:,1], c=col[y], s=1)
    plt.tight_layout()
    plt.show()
    # plt.rcParams['savefig.dpi'] = 300 #图片像素
    # plt.savefig(r"C:\Users\admin\Desktop\tSNE_high_res.png")
    # ax = fig.add_subplot(111)
    # handle_ptr_Lst  = []
    # for y_idx in range(len(code_name_Lst)-1):
    #     range_Lst_i = [instance_num_1pers*y_idx, instance_num_1pers*(y_idx +1)]
    #     sct_i = ax.scatter(Z[range_Lst_i,0], Z[range_Lst_i,1], c=col[y[range_Lst_i]], s=1)
    #     handle_ptr_Lst.append(sct_i)
    # fig.tight_layout()
    # #for y_idx in len(code_name_Lst):
    # plt.legend(handles = handle_ptr_Lst, labels = [str(i) for i in range(len(code_name_Lst))], loc = 'best') # loc='upper right'
    # plt.show()
    # y[0:instance_num_1pers] = 0; y[instance_num_1pers:2*instance_num_1pers] = 1; 