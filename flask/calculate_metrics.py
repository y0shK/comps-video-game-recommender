"""
Calculate and obtain metrics to assess recommendation
"""

def calculate_confusion_matrix(cos_sims, thresholds, y_true):

    tp_tot = 0
    fp_tot = 0
    fn_tot = 0
    tn_tot = 0
    fp_tp_pairs = []

    for t in thresholds:
        # use cosine similarity as probability
        y_pred = [1 if i > t else 0 for i in cos_sims]

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y_true[i] == 1:
                tp += 1
            elif y_pred[i] == 1 and y_true[i] == 0:
                fp += 1
            elif y_pred[i] == 0 and y_true[i] == 1:
                fn += 1
            elif y_pred[i] == 0 and y_true[i] == 0:
                tn += 1

        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        tn_tot += tn

        # pdb.set_trace()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        print((fpr, tpr, t))
        fp_tp_pairs.append((fpr, tpr, t))
        #print(fp_tp_pairs)

    return fp_tp_pairs

def calculate_average_pairs(pairs):
    # take the average of several games ROC to see what it looks like

    total_fprs_tprs = pairs

    fvals = [0] * len(total_fprs_tprs[0])
    tvals = [0] * len(total_fprs_tprs[0])
    threshold_vals = [0] * len(total_fprs_tprs[0])

    # pdb.set_trace()

    # sum all TPR and FPR values on a per game, per tuple basis
    for i in range(len(total_fprs_tprs)):
        for j in range(len(total_fprs_tprs[i])):
            fvals[j] += total_fprs_tprs[i][j][0]
            tvals[j] += total_fprs_tprs[i][j][1]
            threshold_vals[j] += total_fprs_tprs[i][j][2]
            #print("threshold ", total_fprs_tprs[i][j][0], total_fprs_tprs[i][j][1], total_fprs_tprs[i][j][2])

    # pdb.set_trace()

    # then take the average
    avg_fvals = [fvals[j] / len(total_fprs_tprs) for j in range(len(fvals))]
    avg_tvals = [tvals[j] / len(total_fprs_tprs) for j in range(len(tvals))]
    avg_thresholds = [threshold_vals[j] / len(threshold_vals) for j in range(len(threshold_vals))]

    return [avg_fvals, avg_tvals, avg_thresholds]
