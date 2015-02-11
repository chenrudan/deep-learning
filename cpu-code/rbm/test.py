import numpy


def test():
	
    f=open('weight_py.txt','a')
    if f==None:
        print 'file cannot open'
                #   f.write(str(rbm.W.get_value()))
    f.write('hello')
    f.close()

if __name__ == '__main__':
	test() 
