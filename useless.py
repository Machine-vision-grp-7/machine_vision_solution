#%% deal with results
L_work = L_results.copy()
for i in range(len(L_work)):
    L_work[i] = [L_work[i][0],L_work[i][1],round(L_work[i][2][3],3)]
    print(L_work[i])
#%% best ones
L_results_sorted = sorted(L_results, key = lambda x: x[2][3])
bests = L_results_sorted[:3]
for i in  bests:
    print(i)

# %%
plt.figure(figsize=(10,6))
AVE = sum(L_truth)/len(L_truth)
plt.title("Number of apples per picture in the detection dataset")
plt.xlabel("Number of apple")
plt.ylabel("Freqency")
plt.text(50,20,"Average number of apple per picture = " + str(round(AVE,2)))
plt.bar(L_etiquette,L_height)



#%%
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

trilist1 = L_truth.copy()
list2 = L_found.copy()
trilist1, list2 = zip(*sorted(zip(trilist1, list2)))

matplotlib.rc('font', **font)

plt.figure(figsize=(15,10))
plt.title("R2 score for the traditional approach (over 570 samples)")
plt.ylabel("Predicted values")
plt.xlabel("Actual values")
#for i_point in range(len(list2)):
#            plt.plot([i_point,i_point],[trilist1[i_point],list2[i_point]],c="r",label="predicted value")
plt.text(10,70,"R2 = "+str(round(R2,5)))
plt.text(10,65,"normalized RSS = 392.6")
plt.scatter(trilist1,list2,c="royalblue",marker="o",label="predicted value")
plt.plot([0,max(trilist1)],[0,max(trilist1)],c="red",linewidth=3.0)

#plt.ylim(-30,30)
plt.show()
#%%
L_thibo1 = [4, 5, 5, 5, 6, 2, 4, 5, 5, 6, 6, 8, 8, 9, 9, 9, 10, 11, 6, 7, 7, 8, 8, 8, 8, 9, 10, 11, 16, 7, 7, 8, 10, 10, 11, 11, 12, 12, 13, 8, 9, 11, 11, 12, 12, 8, 16, 12, 14, 14, 16, 19, 12, 15, 17, 14, 21, 24, 18, 25, 18, 19, 19, 20, 21, 23, 17, 20, 23, 25, 25, 25, 19, 20, 20, 21, 22, 22, 23, 21, 21, 28, 21, 26, 29, 29, 31, 24, 27, 34, 34, 27, 28, 34, 34, 29, 33, 34, 39, 28, 31, 34, 36, 31, 32, 33, 34, 38, 47, 32, 34, 35, 37, 37, 35, 34, 38, 38, 40, 42, 36, 37, 39, 47, 35, 37, 39, 41, 46, 39, 40, 43, 44, 44, 44, 45, 46, 53, 40, 51, 54, 40, 44, 44, 46, 47, 48, 49, 58, 34, 43, 46, 49, 49, 50, 51, 35, 40, 46, 46, 49, 53, 44, 45, 51, 48, 52, 56, 36, 41, 43, 46, 49, 50, 33, 41, 43, 45, 45, 46, 46, 48, 53, 56, 42, 47, 48, 50, 45, 46, 49, 51, 59, 42, 53, 53, 53, 57, 58, 46, 51, 40, 53, 54, 57, 58, 60, 60, 62, 43, 50, 61, 63, 46, 59, 56, 64, 67, 67, 75, 68, 68, 72, 77, 72, 71, 72, 77, 77, 79, 82, 71, 78, 80, 97, 97, 70, 89, 69, 89, 101, 94, 70, 79, 89, 82, 84, 93, 94, 81, 82, 79, 83, 73, 85, 88, 92, 112, 88, 90, 91, 83, 93, 87, 108, 98, 91, 117, 109, 112]
L_thibo2 = [3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 13, 14, 14, 14, 14, 15, 15, 15, 16, 17, 17, 18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 47, 47, 48, 48, 48, 49, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 54, 58, 58, 59, 59, 60, 60, 61, 65, 66, 67, 67, 68, 69, 69, 69, 70, 70, 70, 70, 70, 73, 73, 74, 75, 76, 77, 78, 78, 78, 79, 79, 79, 79, 80, 80, 81, 81, 82, 82, 85, 85, 85, 87, 88, 89, 90, 91, 95, 95, 98, 100, 100, 105, 107]
plt.figure(figsize=(15,10))

S_yi_minus_ychapi_squaredTIBO = 0
for i in range(len(L_thibo2)):
    S_yi_minus_ychapi_squaredTIBO+= (L_thibo2[i]-L_thibo1[i])**2
print(S_yi_minus_ychapi_squaredTIBO)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

plt.title("R2 score for the machine learning approach (over 270 samples)")
plt.ylabel("Predicted values")
plt.xlabel("Actual values")
#for i_point in range(len(list2)):
#            plt.plot([i_point,i_point],[trilist1[i_point],list2[i_point]],c="r",label="predicted value")
plt.text(10,70,"R2 = "+str(round(0.900164390506853,5)))
plt.text(10,65,"normalized RSS = "+str(round(15864/270,1)))
plt.scatter(L_thibo2,L_thibo1,c="royalblue",marker="o",label="predicted value")
plt.plot([0,max(L_thibo1)],[0,max(L_thibo1)],c="red",linewidth=3.0)
#plt.ylim(-30,30)
plt.show()
# %%
import seaborn

y_true= L_all_truth
plt.xticks(np.arange(3,80,10))
fig = plt.figure(figsize=(10,10))
train_counts= seaborn.countplot(y_true)
plt.xticks(np.arange(min(y_true),max(y_true),10), np.arange(min(y_true),max(y_true),10))
# %%
from collections import Counter

def histogram(iterable, low, high, bins):
    '''Count elements from the iterable into evenly spaced bins

        >>> scores = [82, 85, 90, 91, 70, 87, 45]
        >>> histogram(scores, 0, 100, 10)
        [0, 0, 0, 0, 1, 0, 0, 1, 3, 2]

    '''
    step = (high - low + 0.0) / bins
    dist = Counter((float(x) - low) // step for x in iterable)
    L_etiquette = ["["+str(round(i))+","+str(round(i+step))+"]" for i in np.linspace(low,high,bins)]
    return [dist[b] for b in range(bins)],L_etiquette

nb = 20
L_histocool,L_etiquette = histogram(L_all_truth, min(L_all_truth), max(L_all_truth), nb)


# %%

plt.figure(figsize=(20,10))
plt.tick_params(axis='both', which='major', labelsize=10)
plt.bar(L_etiquette,L_histocool)

# %%
