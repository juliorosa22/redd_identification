from basicLib import *
import matplotlib.pyplot as plt
model='dense_main1'
roc_file=curr_work_dir+f'/roc_curve_{model}.h5'
data=h5.File(roc_file,'r')

fpr_micro=data['fpr_micro'][:]
fpr_macro=data['fpr_macro'][:]
tpr_micro=data['tpr_micro'][:]
tpr_macro=data['tpr_macro'][:]
roc_micro=data['roc_micro_macro'][0]
roc_macro=data['roc_micro_macro'][1]
print(roc_micro)
print(roc_macro)

lw = 4

# Plot all ROC curves
plt.figure()
plt.plot(fpr_micro, tpr_micro,
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_micro),
         color='deeppink', linestyle=':', linewidth=lw)

plt.plot(fpr_macro, tpr_macro,
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_macro),
         color='navy', linestyle=':', linewidth=lw)

#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#for i, color in zip(range(num_classes), colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
#             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for: {model}')
plt.legend(loc="lower right")
plt.show()

# np.savetxt(f'roc_fprmicro_{model}',fpr_micro, delimiter='.')
# np.savetxt(f'roc_fprmacro_{model}',fpr_macro, delimiter='.')
# np.savetxt(f'roc_tprmicro_{model}',tpr_micro, delimiter='.')
# np.savetxt(f'roc_tprmacro_{model}',tpr_macro, delimiter='.')