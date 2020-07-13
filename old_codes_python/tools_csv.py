from basicLib import *

Simu_Signals_path = '/home/julio/Framework_Projeto/Data_sets/Simu_Signals'
Redd_path = '/home/julio/Framework_Projeto/Data_sets/REDD'

def get_apliance_name(dir_path):
    aux = dir_path.split('/')
    n=len(aux)
    return aux[n-2:]


def get_REDD_apliance_name(dir_path,file_name):
    aux=file_name.split('/')
    index=aux[len(aux)-1]
    index=(index.split('_'))[1].split('.')
    i=int(index[0])
    f = open(dir_path+"/"+aux[len(aux)-2]+"/labels.dat",'r')
    l=[line for line in f]
    f.close()
    return str(aux[len(aux)-2]+": "+l[i-1])

def get_REDD_P_array(file_path):
    if  Redd_path not in file_path:
        file_path = Redd_path + file_path
    f = open(file_path,'r')
    time_v=[]
    power_v=[]
    aux=[]
    #faz a leitura dos timestamps e das potencias
    for line in f:
        aux = line.split(' ')
        aux[1]=aux[1][0:len(aux[1])-1]
        time_v.append(int(aux[0]))
        power_v.append(float(aux[1]))
    f.close()
    power_array = np.array(power_v,dtype = np.float32)
    return power_array

def getLowFreqData(file_path):
    if  Redd_path not in file_path:
        file_path = Redd_path + file_path
    sig=[]
    ts=[]
    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            t=int(row[0].split('.')[0])
            s=float(row[1])
            #print(s,t)
            ts.append(t)
            sig.append(s)
    sig_M = np.array(sig,dtype=np.float32)
    ts_M=np.array(ts,dtype=np.int32)
    return ts_M,sig_M

def getHighFreqData(file_path):
    if  Redd_path not in file_path:
        file_path = Redd_path + file_path

    lv = np.empty((1,275),dtype=np.float64)
    linf=np.empty((1,2),dtype=np.int32)

    c=0
    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print(c)
            inf=[float(row[0]),float(row[1]) ]
            inf = np.array([inf],dtype=np.float64)
            linf= np.append(linf,inf,0)
            lv=np.append(lv,np.array([row[2:]],dtype=np.float64),0)

            c=c+1


    return linf[1:],lv[1:]

def getHighTimestamp(file_path):

    if  Redd_path not in file_path:
        file_path = Redd_path + file_path

    lt=[]
    lc=[]
    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:

            lt.append(int(row[0][0:10]))

            lc.append(int(row[1].split('.')[0]))
    vt = np.array(lt,dtype=np.int64)
    vc=np.array(lc,dtype=np.int32)
    return vt,vc

def get_P_power_array(csv_file_path):
    if Simu_Signals_path not in csv_file_path:
        csv_file_path = Simu_Signals_path+csv_file_path
    f = open(csv_file_path,'r')
    csv_file = csv.reader(f)
    l=[]
    for row in csv_file:
        l.append(row)
    f.close()
    N=len(l[0])
    list_P=[]#lista com as potencias ativas da phase A
    list_P.append([l[0][i] for i in range(N) if i%2==0])

    phase_A_data= np.array(list_P,dtype=np.float32)
    return phase_A_data



def getSimuValues_arrays(csv_file_path):#funcao que pega todos os valores simulados de uma unica casa

    if Simu_Signals_path not in csv_file_path:
        csv_file_path = Simu_Signals_path+csv_file_path

    f = open(csv_file_path,'r')
    csv_file = csv.reader(f)
    l=[]
    for row in csv_file:
        l.append(row)
    f.close()
    N=len(l[0])
    real_data=[]
    img_data=[]
    real_data.append([l[0][i] for i in range(N) if i%2==0])
    real_data.append([l[1][i] for i in range(N) if i%2==0])

    img_data.append([l[0][i] for i in range(N) if i%2!=0])
    img_data.append([l[1][i] for i in range(N) if i%2!=0])

    real_array=np.array(real_data,dtype=np.float32)
    img_array=np.array(img_data,dtype=np.float32)
    t=np.linspace(0,(N/(2*3600)),num=N/2)
    return real_array,img_array,t


def mixSignals(signal_list, weights):
	""" Return a sound array mixed in proportion with the ratios given by weights"""
	mixSignal = np.zeros(len(signal_list[0]))
	i = 0
	for signal in signal_list:
		mixSignal += signal*weights[i]
		i += 1

	return mixSignal
