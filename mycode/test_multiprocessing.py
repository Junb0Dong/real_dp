import multiprocessing as mp
 
count = 0
 
class MultiOneProcess(mp.Process):
    def __init__(self,name):
        super().__init__()
        self.name = name
    def run(self) -> None:
        global count
        count += 1
        print('process name %s is running----count:%d'%(self.name, count))
 
if __name__ == '__main__':
    p_list = []
    for i in range(10):
        name = 'process_%d'%i
        p = MultiOneProcess(name = name)
        p.start()
        p_list.append(p)
 
    for p in p_list:
        p.join()
    print('this main process')