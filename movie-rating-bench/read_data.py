import pandas as pd
import os
import re
rootdir0="/home/launch/mhe/bench/movie-rating-bench/data/review_polarity/txt_sentoken/neg/"
rootdir1="/home/launch/mhe/bench/movie-rating-bench/data/review_polarity/txt_sentoken/pos/"
count=0
dic={}
data=[]
for root, dirnames, filenames in os.walk( rootdir0 ):
#    print 'if ".txt" in filename:'
    for filename in filenames:
	count=count+1
#	print count
	#print "filename="
	#print filename
        #id.append(filename)
	myfile=open(os.path.join(root,filename))
	review0=myfile.read()
	review=re.sub(r'\n',' ',review0)
#	print review
	dic={'id':count,'sentiment':0,'review':review}
	data.append(dic)
count=0
for root, dirnames, filenames in os.walk( rootdir1 ):
#    print 'if ".txt" in filename:'
    for filename in filenames:
        count=count+1
#	print "dir1"
#	print count
        myfile=open(os.path.join(root,filename))
        review0=myfile.read()
	review=re.sub(r'\n',' ',review0) 
        dic={'id':count,'sentiment':1,'review':review}
        data.append(dic)

ordered=sorted(data)
output = pd.DataFrame(ordered)

output.to_csv("bench_labeled_review.tsv", sep='\t', index=False)
