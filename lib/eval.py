def f1_score(prec,recall):
    return 2*prec*recall/(prec + recall)

def eval(predicted, ans,precAt=None):
    if precAt is not None and len(predicted)!=0:
        predicted = [predicted[precAt-1]]
    if len(predicted) == 0 and len(ans) == 0:
        return 1,1,1
    elif len(predicted) == 0 and len(ans) != 0:
        return 0,0,0
    elif len(predicted) != 0 and len(ans) == 0:
        return 0,0,0
    else:
        predicted_10 = predicted[:min(10,len(predicted))]
        correctat10 = 0
        correct = 0

        for i in predicted_10:
            if i in ans:
                correctat10 += 1

        for i in predicted:
            if i in ans:
                correct += 1
        prec = correct/len(predicted)
        prec10 = correctat10/min(len(predicted_10),len(ans))
        recall = correct/len(ans)
        return prec, recall, prec10


