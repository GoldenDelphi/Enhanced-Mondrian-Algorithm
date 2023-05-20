import anonypy
import datetime
import pandas as pd
from multiprocessing import Process, Pipe
from generate_fake_dataset import dataset

def pprint(self, t):
    for i in t:
        print(i)

# read dataset from file
def read_dataset(self, filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(line.split(","))
    return dataset

class MultiprocessDataSpliting():
    def __init__(self, K: int, distance_fn):
        self.K = K
        self.distance_fn = distance_fn

    # divide the cluster into two clusters based on the centroids
    def splitCurrentCluster(self, cluster):
        centroid = [None, None]
        # find the two farthest most records in the cluster and use them as centroids
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if centroid == [None, None] or self.distance_fn(cluster[i], cluster[j]) > self.distance_fn(
                    centroid[0], centroid[1]
                ):
                    centroid = [cluster[i], cluster[j]]

        # divide the cluster into two clusters based on the centroids
        # TODO: Deal with outliers
        new_clusters = [[], []]
        for i in range(len(cluster)):
            if self.distance_fn(cluster[i], centroid[0]) < self.distance_fn(cluster[i], centroid[1]):
                new_clusters[0].append(cluster[i])
            else:
                new_clusters[1].append(cluster[i])

        for i in range(2):
            if (len(new_clusters[i]) == 1):
                new_clusters[i].pop(0)
        return new_clusters


    # use divide and conquer approach for splitting the dataset into parallel number of clusters
    def splitCluster(self, dataset, parent_send_end = None):
        # print(len(dataset))
        if (len(dataset) < 2 * self.K):
            if parent_send_end:
                parent_send_end.send([dataset])
                return
            else:
                return [dataset]
        
        clusters = self.splitCurrentCluster(dataset)
        pipes = [Pipe(False) for _ in range(2)]
        processes = [Process(target=self.splitCluster, args=(cluster, send_end))
                    for cluster, (recv_end, send_end) in zip(clusters, pipes)]

        processes[0].start()
        processes[1].start()

        processes[0].join()
        processes[1].join()

        result = []
        for recv_end, send_end in pipes:
            result.extend(recv_end.recv())

        if parent_send_end:
            # print(result)
            parent_send_end.send(result)
        else:
            return result

class ClusterAnonymizer():
    def __init__(self, columns, feature_columns, sensitive_column, K):
        self.columns = columns
        self.feature_columns = feature_columns
        self.sensitive_column = sensitive_column
        self.K = K

    def anonimize_current_cluster(self, cluster, send_end):
        df = pd.DataFrame(data=cluster, columns=self.columns)
        df['gender'] = df['gender'].astype('category')
        p = anonypy.Preserver(df, self.feature_columns, self.sensitive_column)
        rows = p.anonymize_k_anonymity(k=self.K)

        send_end.send(pd.DataFrame(rows, columns=self.columns))
    
    def anonimize(self, clustered_data):
        anonymized_dataframe = pd.DataFrame(data=[], columns=self.columns)
        processes = []
        pipes = []
        for i in range(len(clustered_data)):
            if (len(clustered_data[i]) != 0):
                pipe = Pipe(False)
                pipes.append(pipe)
                _, send_end = pipe
                process = Process(target=self.anonimize_current_cluster, args=(clustered_data[i],send_end))
                processes.append(process)
                process.start()
        
        for process in processes:
            process.join()
        
        for recv_end, send_end in pipes:
            anonymized_dataframe = pd.concat([anonymized_dataframe, recv_end.recv()])
        anonymized_dataframe.reset_index(drop=True, inplace = True)

        return anonymized_dataframe

K = 4  # k-anonymity

# sample dataset
columns, feature_columns, sensitive_column, data = dataset()

# create a custom distance function for figuring out the distance
def distance_fn(rec1, rec2):
    return (abs(rec1[0] - rec2[0]) + [10, 0][rec1[1] == rec2[1]] + abs(rec1[2] - rec2[2]) * 0.001)

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    dataSplitter = MultiprocessDataSpliting(K, distance_fn)
    clusters = dataSplitter.splitCluster(data)
    # print(clusters)
    end_time = datetime.datetime.now() 
    tdelta = end_time - start_time
    print("Clustering time taken:", tdelta)
    
    cluster_anonymizer = ClusterAnonymizer(columns, feature_columns, sensitive_column, K)
    anonymized_dataframe = cluster_anonymizer.anonimize(clusters)
    end_time = datetime.datetime.now() 
    tdelta = end_time - start_time
    
    print("Total time taken:", tdelta)
    print('#'*43 + '\nTop 30 rows of the dataset:\n', anonymized_dataframe.head(30))
    print('#'*43 + '\nDealing with Catagorical Values\n', anonymized_dataframe[(anonymized_dataframe['gender'] != 'M') & (anonymized_dataframe['gender'] != 'F')])
    print('#'*43 + '\nFinal Dataset\n', anonymized_dataframe)
