# Konfigurasi Spark
import os
import sys

# 1. Mengeset variabel yang menyimpan lokasi di mana Spark diinstal
# spark_path = "D:/spark-2.1.0-bin-hadoop2.7"
spark_path = "D:/its/big_data/spark/"

# 2. Menentukan environment variable SPARK_HOME
os.environ['SPARK_HOME'] = spark_path

# 3. Simpan lokasi winutils.exe sebagai environment variable HADOOP_HOME
os.environ['HADOOP_HOME'] = spark_path

# 4. Lokasi Python yang dijalankan --> punya Anaconda
#    Apabila Python yang diinstall hanya Anaconda, maka tidak perlu menjalankan baris ini.
os.environ['PYSPARK_PYTHON'] = sys.executable

# 5. Konfigurasi path library PySpark
sys.path.append(spark_path + "/bin")
sys.path.append(spark_path + "/python")
sys.path.append(spark_path + "/python/pyspark/")
sys.path.append(spark_path + "/python/lib")
sys.path.append(spark_path + "/python/lib/pyspark.zip")
sys.path.append(spark_path + "/python/lib/py4j-0.10.4-src.zip")

# 6. Import library Spark
#    Dua library yang WAJIB di-import adalah **SparkContext** dan **SparkConf**.
from pyspark import SparkContext
from pyspark import SparkConf
import numpy as np

# Setting konfigurasi (opsional)
conf = SparkConf()
conf.set("spark.executor.memory", "2g")
conf.set("spark.cores.max", "4")

sc = SparkContext("local", conf=conf)
#    Apabila berhasil, maka ketika sc di-print akan mengeluarkan nilai <pyspark.context.SparkContext object>
print sc

from numpy import array

from pyspark.mllib.clustering import KMeans, KMeansModel

data = sc.textFile("complete.csv")

header = data.first()
data=data.filter(lambda line: line != header)


from collections import namedtuple
import matplotlib.pyplot as plt
from math import sqrt

def RepresentsFloat(s, r):
    try: 
        float(s)
        float(r)
        return True
    except ValueError:
        return False
fields = ('latitude','longitude')

Ufo = namedtuple('Ufo',fields,verbose=True)

def parse(row):
    row[9] = float(row[9])
    row[10]= float(row[10])
    return Ufo(*row[9:11])
    
temp_data = data.map(lambda x: x.split(','))
temp_data = temp_data.filter(lambda x: len(x) == 12)
temp_data = temp_data.filter(lambda x: RepresentsFloat(x[9],x[10])==True )
temp_data = temp_data.filter(lambda x: (x[9]!='0' or x[10]!='0'))
datafix = temp_data.map(parse)

def plot(centers):
    for i in range(len(centers)):
        plt.plot(centers[i], '-o')
   
k=5     

parsedData = datafix.map(lambda line: array([line.latitude, line.longitude]))
clusters = KMeans.train(parsedData, k, maxIterations=1000, initializationMode="random")

cmap = plt.get_cmap('jet')
colormap = cmap(np.linspace(0, 1, k))

a=clusters.predict(parsedData)
b=parsedData.collect()        
c=np.mat(b)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(c[:,0], c[:,1], c=colormap[a.collect()], s=20)  
plt.show

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))