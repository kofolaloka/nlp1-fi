#!/usr/bin/env python2

try:
    import sys
    import os
    import os.path
    import urllib
except Exception as e:
    print "exception %s (did you run install_packages.sh ?)"%str(e)

url = "http://commondatastorage.googleapis.com/books/syntactic-ngrams/eng/unlex-verbargs.%02d-of-99.gz"
nums = xrange(0,99)

def _dir():
    dirname = os.path.normpath(sys.argv[1])
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except:
            print "unable to create directory %s"%dirname
    try:
        print "cd to %s.."%dirname
        os.chdir(dirname)
    except Exception as e:
        print "unable to cd to directory:%s"%str(e)
        sys.exit(-1)

    return dirname

def main():
    assert len(sys.argv) > 1, "usage: %s dirname"%sys.argv[0]
    _dir()
    
    for num in nums:
        curr_url = url%num
        filename = (curr_url).split("/")[-1]

        print "downloading %s to %s"%(curr_url,filename)
        try:
            urllib.urlretrieve(curr_url, filename)
        except Exception as e:
            print "unable to retrieve %s : %s"%(curr_url,str(e))
            sys.exit(-1)
    print "all done."

if __name__=="__main__":
    main()

