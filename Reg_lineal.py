#importamos las librearias que vamos a utilizar import numpy as np
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn

#Descargamos los datos de nuestro csv
df = pd.read_csv('Student_Performance.csv')

#Dado que nuestra columna de actividades extracurriculares tiene valores de string los cambiamos a valores numericos de 0 y 1 en caso de que no tengan o si tengan actividades extracurriculares
df['Extracurricular Activities']=df['Extracurricular Activities'].replace('Yes',1)
df['Extracurricular Activities']=df['Extracurricular Activities'].replace('No',0)

#Dividimos nuestra base de datos en dos apartados, el de train que sera con el que entrenaremos el modelo y el de test que sera con el cual comprobaremos el rendimiento del mismo 
df_train=df[:9000]
df_test=df[9000:]

#Parado poder visualizar de mejor forma nuestras graficas ordenaremos el data set de test de menor a mayor segun el inide de desempeño de los alumnos 
df_test.sort_values("Performance Index")

#Divimos nuestrso dos DataSets (trai y test) en nuetsras variables x (nuetras variables independientes ) y y (nuestra variable dependiente es decir la variable a predecir)
x_train=df_train[['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']]
y_train=df_train[["Performance Index"]]

x_test=df_test[['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']]
y_test=df_test[["Performance Index"]]

__errors__= [];  #Variable global donde guardaremos nuetsros errores/loss para visualizarlos mas adelante 


#Funcion para obtener la hipotesis con nuestrso parametros(params) donde sample son los valores de x que vamos a utilizar 
def h(params,sample):
  acum= np.dot(params,sample)
  return(acum)


#La funcion de error es donde obtendremos nuestro MSE de los datos 
def show_errors(params, samples,y):
  global __errors__
  error_acum =0
  hyp=[]
  for i in range(len(samples)):
    val = h(params,samples.loc[i])
    hyp.append(val)
  hyp=np.array(hyp)

#Mostrar los datos
  res=pd.DataFrame(pd.Series(hyp),columns=['Hyp'])
  res['Y']=y['Performance Index']
  print(res)

  error=hyp-y['Performance Index']
  error=error ** 2 
  error_acum=error.sum()
  mean_error_param=error_acum/len(samples)
  __errors__.append(mean_error_param) #Guardamos el error para despues graficarlo 


#Funcion de optimizacion de nuestro gradiente descendiente donde actualizamos nuestros parametros por casa renglon de nuesro DF 
def GD(params, samples, y, alfa):
  temp = params
  for j in range(len(params)):
    acum =0
    for i in range(len(samples)):
      error = h(params,samples.loc[i]) - y.loc[i]['Performance Index']
      fila=samples.loc[i]
      acum = acum + error*fila[j]  #Sumatory part of the Gradient Descent formula for linear Regression.
      temp[j] = params[j] - alfa*(1/len(samples))*acum  #Subtraction of original parameter value with learning rate included.
  return temp

#Funcion de normalización que nos permitira combertir los valores de nuestro DF en un rango de 0 a 1 para que no dispersos los datos de nuetsras variables x
def norma(samples):
	#X_new = (X - X_min)/(X_max - X_min)
	name_col=samples.columns.values
	for i in name_col:
		max_val = samples[i].max()
		min_val = samples[i].min()
		samples[i] = (samples[i]-min_val)/(max_val-min_val)
	return samples


#Iniciamos nuetsros parametros en 0 y el sample que seran nuestro DF con los diferentes valor de nuestras x y nuetsra variable de respuesta y 
params = [0,0,0,0,0,0]
samples =norma(x_train)
y = y_train

alfa =.01  # Declaramos nuestro learning rate
constante=np.ones((len(samples)))
#Agregar una columna de 1 para evaluar nuestro parametro de b es decir la constante
samples["Val constante"]=constante


epochs = 0

while True:  #  Vamos a correr la funcion de gradiente descendiente hasta que nuestros parametros se optimicen o se cumpla el numero de epocas 
	oldparams = list(params)
	print(params)
	params=GD(params, samples,y,alfa)
	show_errors(params, samples, y)  #Mostramos los errores acumulados para ver como se comporta el modelo 
	print(params)
	epochs = epochs + 1
	if(oldparams == params or epochs>10):   #  si los nuevos parametros son iguales a los viejos o el numero de epocas es mayor a 10 detenemos el ciclo 
		print ("Parametros finales:")
		print (params)
		break


#Mostramos la grafica de los errores para ver como se comportron en cada una de las epocas 
plt.plot(__errors__) 
plt.show()


from numpy.core.fromnumeric import transpose
#predicciónes
#Predicciónes (Para las predicciones solo utilizaremos 35 datos del DF de Test )

p=params
#Agregar una columna de 1 para evaluar nuestro parametro de b es decir la constante
const=np.ones((len(x_test[:35])))
x_test=norma(x_test[:35])  #Normalizamos nuestros datos 
x_test["constante"]=const

#Aplicamos transpuesta para poder adaptar a las dimenciones para la funcion de la hipotesis y la llamamos con los nuevos parametros 
X=np.transpose(x_test) 
yfit=h(p,X)

#Creamos un nuevo DF para guardar los datos optenidos de neustra regresión lineal y los valores reales 
result=pd.DataFrame(yfit, columns=['Hyp Y'])
result['Hyp Y'] = result['Hyp Y']
result['Real Y']=y_test[:35].values

#Graficamos ambos valores para observar como se conportan los datos de predicción con respecto a los reales 
result.plot()
plt.show()