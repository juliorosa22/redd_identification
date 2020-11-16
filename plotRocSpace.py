from basicLib import *
import pandas as pd
import matplotlib.pyplot as plt


main_roc_file=curr_work_dir+'/Models/'+'main2_roc.xlsx'
dev_name='7 refrigerator'
df=pd.read_excel(main_roc_file,sheet_name=dev_name,engine='openpyxl')
models_name=df['model']
labels=['Random guess']
labels.extend(models_name)
print(labels)
tpr=df['true_positive_rate']
fpr=df['false_positive_rate']


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Space -'+dev_name)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.plot([0,1],[0,1],color='navy',linestyle='--')
colors=['red','blue','green']
for i in range(len(tpr)):
    plt.plot(fpr[i],tpr[i],color=colors[i],marker='o')
plt.legend(labels)
plt.show()