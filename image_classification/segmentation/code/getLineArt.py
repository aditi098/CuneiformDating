
import pandas as pd
import requests
import tqdm
import threading
import os

df = pd.read_csv('/trunk/shared/cuneiform/full_data/expanded_catalogue.csv')  
pvalues = df['id']
# pvalues = [1,2,3]

img_base_url = "https://cdli.mpiwg-berlin.mpg.de/dl/lineart/"

def task1():
    for pid in tqdm.tqdm(pvalues[10000:20000]):
        img_name = "P"+str(pid).zfill(6)+"_l.jpg"
        
        if os.path.exists("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name):
            continue
        
        img_url = img_base_url + img_name
        r = requests.get(img_url)
        if r.status_code == 500 or r.status_code == 404:
            continue
        img_data = r.content 
        with open("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name, 'wb') as handler:
            handler.write(img_data)

def task2():
    for pid in tqdm.tqdm(pvalues[30000:40000]):
        img_name = "P"+str(pid).zfill(6)+"_l.jpg"
        
        if os.path.exists("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name):
            continue
        img_url = img_base_url + img_name
        r = requests.get(img_url)
        if r.status_code == 500 or r.status_code == 404:
            continue
        img_data = r.content 
        with open("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name, 'wb') as handler:
            handler.write(img_data)
            
def task3():
    for pid in tqdm.tqdm(pvalues[50000:60000]):
        img_name = "P"+str(pid).zfill(6)+"_l.jpg"
        
        if os.path.exists("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name):
            continue
        
        img_url = img_base_url + img_name
        r = requests.get(img_url)
        if r.status_code == 500 or r.status_code == 404:
            continue
        img_data = r.content 
        with open("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name, 'wb') as handler:
            handler.write(img_data)
            
def task4():
    for pid in tqdm.tqdm(pvalues[70000:80000]):
        img_name = "P"+str(pid).zfill(6)+"_l.jpg"
        
        if os.path.exists("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name):
            continue
        
        img_url = img_base_url + img_name
        r = requests.get(img_url)
        if r.status_code == 500 or r.status_code == 404:
            continue
        img_data = r.content 
        with open("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name, 'wb') as handler:
            handler.write(img_data)

def task5():
    for pid in tqdm.tqdm(pvalues[90000:100000]):
        img_name = "P"+str(pid).zfill(6)+"_l.jpg"
        
        if os.path.exists("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name):
            continue
        
        img_url = img_base_url + img_name
        r = requests.get(img_url)
        if r.status_code == 500 or r.status_code == 404:
            continue
        img_data = r.content 
        with open("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name, 'wb') as handler:
            handler.write(img_data)
            
def task6():
    for pid in tqdm.tqdm(pvalues[110000:120000]):
        img_name = "P"+str(pid).zfill(6)+"_l.jpg"
        
        if os.path.exists("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name):
            continue
        
        img_url = img_base_url + img_name
        r = requests.get(img_url)
        if r.status_code == 500 or r.status_code == 404:
            continue
        img_data = r.content 
        with open("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name, 'wb') as handler:
            handler.write(img_data)
            
def task7():
    for pid in tqdm.tqdm(pvalues[130000:]):
        img_name = "P"+str(pid).zfill(6)+"_l.jpg"
        
        if os.path.exists("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name):
            continue
        
        img_url = img_base_url + img_name
        r = requests.get(img_url)
        if r.status_code == 500 or r.status_code == 404:
            continue
        img_data = r.content 
        with open("/trunk/shared/cuneiform/full_data/lineart_images/"+img_name, 'wb') as handler:
            handler.write(img_data)

if __name__ == "__main__":
    print("ID of process running main program: {}".format(os.getpid()))
 
    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))
 
    # creating threads
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2')
    t3 = threading.Thread(target=task3, name='t3')
    t4 = threading.Thread(target=task4, name='t4')
    t5 = threading.Thread(target=task5, name='t5')
    t6 = threading.Thread(target=task6, name='t6')
    t7 = threading.Thread(target=task7, name='t7')
    
    # starting threads
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    
    # wait until all threads finish
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()