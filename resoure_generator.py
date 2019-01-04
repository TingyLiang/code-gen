'''与k8s交互，返回集群资源信息'''


@staticmethod
def get_cluster(self):
    cluster = {'chief': ['localhost:2222'],
               'ps': ['localhost:2223', 'localhost:2224'],
               'worker': ['localhost:2224', 'localhost:2225']}
    return cluster
