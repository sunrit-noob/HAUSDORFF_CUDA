.PHONY : build reqs install clean
NINJA := $(-v ninja > /NUL)

build : reqs
	python setup.py bdist_wheel

reqs :

ifndef NINJA 
	copy %cp%\ninja C:\Windows\System32\bin
endif 
	pip3 install -r requirements.txt

install :
	pip3 install --upgrade --find-links=dist HAUSDORFF_CUDA

clean :
	-rm -rf build %cp%\dist\* *.egg-info