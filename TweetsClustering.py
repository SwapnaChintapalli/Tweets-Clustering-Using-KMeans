'''
Created on Nov 17, 2018

@author: user
'''
from  __future__ import print_function
import sys
import json
import re
import copy
import math
from pip._vendor.distlib.compat import raw_input




#intial data cleaning and preprocessing using regular expressions

regex_str = [
     r'<[^>]+>', 
    r'(?:@[\w_]+)', 
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", 
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', 
    r"(?:[a-z][a-z'\-_]+[a-z])", 
    r'(?:[\w_]+)', 
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens

# defining the Jaccard distance
def jaccard(a,b):
    inter= list(set(a) & set(b))
    I=len(inter)
    union= list(set(a) | set(b))
    U=len(union)
    return round(1-(float(I)/U),4)



#k-means implementation
def kmeans(id,centroids,terms_all,l,k):
    
    count=0
    for h in range(25):    
        count=count+1
        indices=[id.index(item) for item in centroids]
        cen_txt=[ terms_all[x] for x in indices]
        cluster=[]
        for i in range(l):
            d=[jaccard(terms_all[i],cen_txt[j]) for j in range(k)]
            ans= d.index(min(d))
            cluster.append(ans)        
        #print cluster
        
        
        centroids1=up_date(id,cluster,terms_all,l,k)
        sum=0
        for i in range(k):
            if (centroids1[i]==centroids[i]):
                sum=sum+1
        if (sum==k):
            break;   
        centroids = copy.deepcopy(centroids1)
    
    output(cluster,k,id)
    sse(cluster,centroids,terms_all,k,l)
    f.close()

#output in the form of cluster# and the array of the tweetids in that cluster.

def output(cluster,k,id):
        final=[]
        for i in range(k):    
            final.append([j for j, u in enumerate(cluster) if u == i])    
            t=[x for x in final[i]]
            print(i+1,[id[x] for x in t],file=f)

#computing the sum of squared errors
            
def sse(cluster,centroids,terms_all,k,l):
    indices1=[]
    indices=[id.index(item) for item in centroids]
    cen_txt=[ terms_all[x] for x in indices]
    sum=0
    for i in range(k):    
        indices1.append([j for j, u in enumerate(cluster) if u == i])    
        #print indices1[i]
        t=    [ terms_all[x] for x in indices1[i]]
        for j in range(len(indices1[i])):
            sum=sum + math.pow(jaccard(t[j],cen_txt[i]),2)
    print('SSE',sum, file =f)
            
            
# updatin gthe centroids at every iteration
def up_date(id,cluster,terms_all,l,k):
    indices=[]
    new_centxt_index=[]
    new_centroid=[]
    for i in range(k):
        indices.append([j for j, u in enumerate(cluster) if u == i])
        m=indices[i]
        #print m
        #m gives the indices if the elements of every cluster k
        
        if (len(m) != 0):
            txt=[terms_all[p] for p in m]
            sim =[ [jaccard(txt[i],txt[j]) for j in range(len(m))] for i in range(len(m))]
            #print sim
    #symmetric distance matrix    
            #print [sum(i) for i in sim]
            f1=[sum(i) for i in sim]
            #print f.index(min([sum(i) for i in sim]))
    #lower triangular matrix
            new_centxt_index.append( m[(f1.index(min([sum(i) for i in sim])))]) #index of the point closer to all the other points    
    new_centroid=[id[x] for x in new_centxt_index]
    return new_centroid
    

            

# def main():
#     terms_all=[]
#     id=[]
# # with open(str(sys.argv[3]), 'r') as f:
#     with open("Tweets.json", 'r') as f:
#         for line in f:
#             tweet = json.loads(line)
#             t=tweet['text']
#         
#             tokens = preprocess(t)
#             terms_all.append([term for term in tokens])
#         
#             d=tweet['id']
#             id.append(d)
#         
# 
#     l=len(terms_all)
# # text_file = open(str(sys.argv[2]), "r")
#     text_file = open("InitialSeeds.txt", "r")
#     centroids = text_file.read().split(',')
#     centroids= [x.strip('\n') for x in centroids]
#     centroids= [int(x) for x in centroids]    #ids of centroids
# # k=int(sys.argv[1])
# # k=int(sys.argv[1])
#     k=25
#       
#     kmeans(id,centroids,terms_all,l,k)


if __name__ == '__main__':
    terms_all=[]
    id=[]
# with open(str(sys.argv[3]), 'r') as f:
    user_input = raw_input("tweets-k-means")
    input_list = user_input.split(' ')
    numbers = [x.strip() for x in input_list]
    print(numbers[0])
    print("tweets-k-means")
    
    k = int(input(""))
    initialSeeds = input("")
    tweets = input("")
    outputFile = input("")
    with open(str(tweets), 'r') as f:
        for line in f:
            tweet = json.loads(line)
            t=tweet['text']
        
            tokens = preprocess(t)
            terms_all.append([term for term in tokens])
        
            d=tweet['id']
            id.append(d)
        

    l=len(terms_all)
# text_file = open(str(sys.argv[2]), "r")
    
    text_file = open(str(initialSeeds), "r")
    centroids = text_file.read().split(',')
    centroids= [x.strip('\n') for x in centroids]
    centroids= [int(x) for x in centroids]    #ids of centroids
#     k=int(sys.argv[1])
# k=int(sys.argv[1])
#     k=25
    f = open(str(outputFile), 'w')
    kmeans(id,centroids,terms_all,l,k)

#     main()
